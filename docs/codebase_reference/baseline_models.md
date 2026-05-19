# `baseline_models/` Module Reference

This directory provides implementations of non-RL (Reinforcement Learning) decentralized models, serving as benchmarks to compare against RL techniques. Unlike central approaches (like `centralized_dqn.py`), these agents process observations independently and do not share a centralized policy or memory buffer.

## Dependencies & Imports

```python
import numpy as np
import random
from abc import ABC, abstractmethod
from routerl import get_learning_model
```
- **Standard Library**: `abc` is used to enforce contract compliance in subclass implementations. `random` is utilized for standard non-deterministic behaviors.
- **Third-Party**: `numpy` tracks expected value arrays.
- **External Framework**: `routerl.get_learning_model` is imported to inject baseline algorithms (like the Gawron assignment method) provided externally by the simulation framework.

---

## Data Flow & Architecture

1. **Initialization:** The simulation loops through pre-mutation agents. Upon mutation to an Autonomous Vehicle (or when spawning humans with fixed baselines), `registry.py:get_baseline()` intercepts the requested `model` name from `exp_config.json`.
2. **Knowledge Seeding:** Before instantiation, the simulator retrieves the known free-flow network times for the agent's specific Origin-Destination pair. This data (`initial_knowledge`) is passed into the model constructor so agents have a base estimate of route costs.
3. **Execution Loop:** During every episode, the simulation invokes `<Model>.act(state)` for each active agent, converting internal knowledge into a route index.
4. **Observation/Update Loop:** At episode completion, the agent receives a cost/reward based on the travel time and invokes `<Model>.learn(state, action, reward)`. While learning is typically a no-op for fixed baselines like AON or Random, heuristic baselines (like Gawron) update internal probabilities here.

---

## Deep Function/Class References

### `base.py: BaseLearningModel`
The abstract base class defining the strict API all baseline algorithms must follow.

```python
class BaseLearningModel(ABC):
    @abstractmethod
    def act(self, state) -> None:
        pass

    @abstractmethod
    def learn(self, state, action, reward) -> None:
        pass
```

### `aon.py: AON`
Implements the All-Or-Nothing (AON) assignment principle, acting greedily on initial free-flow knowledge indefinitely.

**Initialization (`__init__`)**:
- **Arguments**: `params (dict)`, `initial_knowledge (list|array)`.
- **Logic**: Casts the initial knowledge (which is passed as the *negative* of the travel time to act as a pseudo-reward) into a NumPy array and stores it as `self.cost`.

**Action Selection (`act`)**:
- **Returns**: `int` (route index).
- **Logic**:
```python
def act(self, state) -> int:
    action = int(np.argmax(self.cost))
    return action
```
Always selects the route with the highest expected reward value (shortest free flow time). The `state` argument is intentionally ignored.

### `random.py: Random`
Selects routes using a uniform random distribution, ignoring all costs.

**Action Selection (`act`)**:
- **Returns**: `int` (route index).
- **Logic**:
```python
def act(self, state) -> int:
    action = random.randint(0, len(self.cost) - 1)
    return action
```
Uses the length of `self.cost` strictly to infer the total number of available actions/routes in the discrete action space, pulling uniformly from `[0, N-1]`.

### `registry.py: get_baseline`
A factory pattern function abstracting model instantiation.

**Arguments**:
- `params (dict)`: Algorithm parameters (must contain `"model"` key).
- `initial_knowledge (Any)`: Initial network costs.

**Returns**:
- Returns an instantiated child of `BaseLearningModel`.

**Internal Logic**:
```python
def get_baseline(params, initial_knowledge):
    model = params["model"]
    if model == "aon":
        return AON(params, initial_knowledge)
    elif model == "random":
        return Random(params, initial_knowledge)
    else:
        try:
            return get_learning_model(params, initial_knowledge)
        except:
            raise ValueError('[MODEL INVALID] Unrecognized model: ' + model)
```
If the requested model string matches internal baselines, it instantiates them locally. Otherwise, it delegates instantiation down to `routerl`, effectively extending the benchmark's baseline support without duplicating code.
