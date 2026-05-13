"""Quick integration test for the new v3.0 systems."""
from auto_healer import healer, self_healing
from vector_memory import KolkataVectorMemory
import os

print("=" * 50)
print("INTEGRATION TEST: Kolkata FF v3.0")
print("=" * 50)

# Test 1: Vector Memory
print("\n--- Test 1: Vector Memory ---")
brain = KolkataVectorMemory(context_size=5, db_path='test_brain.json')
assert brain.get_brain_capacity() == 0
brain.remember([1,2,3,4,5], 7, 3)
brain.remember([2,3,4,5,6], 8, 4)
brain.remember([3,4,5,6,7], 2, 5)
assert brain.get_brain_capacity() == 3
result = brain.query([1,2,3,4,5], 3)
assert 7 in result, f"Expected 7 in results, got {result}"
boost = brain.get_prediction_boost([1,2,3,4,5], 3)
assert boost is not None
assert len(boost) == 10
print("  Vector Memory: PASS")

# Test 2: Auto-Healer
print("\n--- Test 2: Auto-Healer ---")
status = healer.get_status()
assert 'api_key_set' in status
assert 'total_heals' in status
assert 'rate_limit' in status

@self_healing(max_retries=0, fallback_value="RECOVERED")
def crashing_function():
    raise RuntimeError("Simulated scraper crash")

result = crashing_function()
assert result == "RECOVERED", f"Expected RECOVERED, got {result}"
assert healer.get_status()['total_heals'] >= 1
print("  Auto-Healer (with fallback): PASS")
print(f"  Heal log entry: {healer.get_log()[-1]['error_type']}: {healer.get_log()[-1]['error_message'][:50]}")

# Test 3: predict_ml import with brain
print("\n--- Test 3: predict_ml_v2 import ---")
import predict_ml_v2 as predict_ml
assert hasattr(predict_ml, 'brain')
assert isinstance(predict_ml.brain, KolkataVectorMemory)
print(f"  predict_ml brain capacity: {predict_ml.brain.get_brain_capacity()}")
print("  predict_ml import: PASS")

# Clean up test file
if os.path.exists('test_brain.json'):
    os.remove('test_brain.json')
if os.path.exists('heal_log.json'):
    os.remove('heal_log.json')

print("\n" + "=" * 50)
print(">>> ALL INTEGRATION TESTS PASSED! <<<")
print("=" * 50)
