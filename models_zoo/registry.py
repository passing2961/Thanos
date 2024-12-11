from collections import defaultdict

_model_registry = defaultdict(list) # mapping of model names to entrypoint fns


def register_model(framework):
    def innter_decorator(fn):
        # add entries to registry dict/sets
        model_name = fn.__name__
        _model_registry[framework].append(model_name)

        return fn
    
    return innter_decorator

def load_model(args, accel=None):
    import models_zoo.models as models
    
    model_name = args.model_name
    if model_name in list_models("all"):
        model = eval(f"models.{model_name}")(args, accel=accel)
    else:
        return None
    return model

def list_models(framework="all"):
    """
    Return list of available model names, sorted alphabetically
    """
    r = []

    if framework == "all":
        for _, v in _model_registry.items():
            r.extend(v)
    else:
        r = _model_registry[framework]
    
    return sorted(list(r))


