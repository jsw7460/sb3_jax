"""Policies: abstract base class and concrete implementations."""

import os
from typing import Any, Optional, Tuple, Union, Callable, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from offline_baselines_jax.common.type_aliases import Params


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    batch_stats: Union[Params]
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
        **kwargs
    ) -> 'Model':

        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        """
        NOTE:
            Here we unfreeze the parameter. 
            This is because some optimizer classes in optax must receive a dict, not a frozendict, which is annoying.
            https://github.com/deepmind/optax/issues/160
        """
        params = params.unfreeze()

        # Frozendict's 'pop' method does not support default value. So we use get method instead.
        batch_stats = variables.get("batch_stats", None)

        if tx is not None: opt_state = tx.init(params)
        else: opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            params=params,
            batch_stats=batch_stats,
            tx=tx,
            opt_state=opt_state,
            **kwargs
        )

    def __call__(self, *args, **kwargs):
        return self.apply_fn({"params": self.params}, *args, **kwargs)

    def apply_gradient(
        self,
        loss_fn: Optional[Callable[[Params], Any]] = None,
        grads: Optional[Any] = None,
        has_aux: bool = True
    ) -> Union[Tuple['Model', Any], 'Model']:

        assert (loss_fn is not None or grads is not None, 'Either a loss function or grads must be specified.')

        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux: grads, aux = grad_fn(self.params)
            else: grads = grad_fn(self.params)
        else:
            assert (has_aux, 'When grads are provided, expects no aux outputs.')

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        new_model = self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state)

        if has_aux:
            return new_model, aux
        else:
            return new_model

    def save_dict(self, save_path: str) -> Params:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))
        return self.params

    def load_dict(self, load_path: str) -> "Model":
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)

    def save_batch_stats(self, save_path: str) -> Params:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.batch_stats))
        return self.batch_stats

    def load_batch_stats(self, load_path: str) -> "Model":
        with open(load_path, 'rb') as f:
            batch_stats = flax.serialization.from_bytes(self.batch_stats, f.read())
        return self.replace(batch_stats=batch_stats)