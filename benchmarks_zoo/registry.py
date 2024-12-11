from collections import defaultdict

_benchmark_registry = defaultdict(list) # mapping of benchmark names to entrypoint fns


def register_benchmark(name):
    def inner_decorator(fn):
        # add entries to registry dict/sets
        benchmark_name = fn.__name__
        _benchmark_registry[name].append(benchmark_name)

        return fn
    
    return inner_decorator

def load_benchmark(benchmark_name, args):
    import benchmarks_zoo.benchmarks as benchmarks
    
    #benchmark_name = args.benchmark_name
    supported_benchmarks = list_benchmarks("all")
    module_name = supported_benchmarks.index(benchmark_name)
    if module_name is None:
        raise NameError(
            f"Benchmark {benchmark_name} is not supported, "
            f"please select from {list(supported_benchmarks.keys())}"
        )
    
    return eval(f"benchmarks.{benchmark_name}")(args)

def get_benchmark_types():
    return list(_benchmark_registry.keys())

def list_benchmarks(framework="all"):
    """Return list of available benchmark names, sorted alphabetically"""
    r = []
    if framework == "all":
        for _, v in _benchmark_registry.items():
            r.extend(v)
    else:
        r = _benchmark_registry[framework]
    if isinstance(r, str):
        return [r]
    return sorted(list(r))