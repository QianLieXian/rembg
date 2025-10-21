import importlib
import os
from typing import Optional, Type

import onnxruntime as ort

from .sessions import (
    get_session_class,
    normalize_session_name,
    sessions,
    sessions_names,
)
from .sessions.base import BaseSession
from .sessions.u2net import U2netSession


def new_session(model_name: str = "u2net", *args, **kwargs) -> BaseSession:
    """
    Create a new session object based on the specified model name.

    This function resolves the session class registered for the requested model
    name (including known aliases) and creates an instance with the provided
    arguments.
    The 'sess_opts' object is created using the 'ort.SessionOptions()' constructor.
    If the 'OMP_NUM_THREADS' environment variable is set, the 'inter_op_num_threads' option of 'sess_opts' is set to its value.

    Parameters:
        model_name (str): The name of the model.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Raises:
        ValueError: If no session class with the given `model_name` is found.

    Returns:
        BaseSession: The created session object.
    """
    normalized_model = normalize_session_name(model_name)
    session_class: Optional[Type[BaseSession]] = get_session_class(normalized_model)

    if session_class is None:
        # Attempt a lazy import for models that might not have been registered
        # yet.  This keeps backwards compatibility with environments where the
        # module providing the session class is available but wasn't imported
        # during package initialisation (for example because an older cached
        # module list is still in use).
        fallback_targets = {
            "bria-rmbg14": "rembg.sessions.bria_rmbg14.BriaRmBg14Session",
        }

        target = fallback_targets.get(normalized_model)
        if target:
            module_name, class_name = target.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                session_class = getattr(module, class_name)
            except (ImportError, AttributeError) as exc:  # pragma: no cover - best effort fallback
                raise ValueError(
                    f"Model '{model_name}' is not available in the current installation."
                ) from exc
            else:
                sessions[normalized_model] = session_class

    if session_class is None:
        available = ", ".join(sorted(sessions_names))
        raise ValueError(
            f"No session class found for model '{model_name}'. Available models: {available}"
        )

    sess_opts = ort.SessionOptions()

    if "OMP_NUM_THREADS" in os.environ:
        threads = int(os.environ["OMP_NUM_THREADS"])
        sess_opts.inter_op_num_threads = threads
        sess_opts.intra_op_num_threads = threads

    return session_class(normalized_model, sess_opts, *args, **kwargs)
