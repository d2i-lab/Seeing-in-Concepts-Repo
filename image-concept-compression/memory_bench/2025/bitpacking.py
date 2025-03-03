import sys
import random
import array
import time
import numpy as np
from pympler import asizeof

import numpy as np
import sys
from pympler import asizeof

class NumpyBitpackedDict:
    def __init__(self, initial_size=1024, load_factor=0.7):
        self.size = initial_size
        self.used = 0
        self.keys = np.zeros(self.size, dtype=np.uint32)
        self.values = np.array([None] * self.size, dtype=object)
        self.PRIME1 = self._next_prime(self.size // 2 - 1)
        self.load_factor = load_factor
        print(self.PRIME1, self.size)

    def _next_prime(self, n):
        def is_prime(k):
            if k < 2: return False
            for i in range(2, int(k**0.5) + 1):
                if k % i == 0: return False
            return True
        while not is_prime(n):
            n += 1
        return n

    def __contains__(self, key):
        packed_key = self._pack_key(*key)
        index = self._find_slot(packed_key)
        return self.values[index] is not None

    def __setitem__(self, key, value):
        packed_key = self._pack_key(*key)
        index = self._find_slot(packed_key)
        
        if self.values[index] is None:
            self.used += 1
            
        self.keys[index] = packed_key
        self.values[index] = value
        
        if self.used > self.load_factor * self.size:
            self._resize()

    def __getitem__(self, key):
        packed_key = self._pack_key(*key)
        index = self._find_slot(packed_key)
        
        if self.values[index] is None:
            raise KeyError(key)
        
        return self.values[index]

    # TODO: Let bitsplit of img_id and concept_id be configurable
    def _pack_key(self, img_id, concept_id):
        if img_id < 0 or img_id >= (1 << 22):
            raise ValueError("img_id must be between 0 and 2^22 - 1")
        if concept_id < 0 or concept_id >= (1 << 10):
            raise ValueError("concept_id must be between 0 and 2^10 - 1")
        return np.uint32((img_id << 10) | concept_id)
    
    def _hash1(self, key):
        return key % self.size

    def _hash2(self, key):
        return self.PRIME1 - (key % self.PRIME1)

    def _find_slot(self, packed_key):
        index = self._hash1(packed_key)
        step = 2*self._hash2(packed_key)+1 # Always odd (coprime to size which is always power of 2)

        # print(f"Finding slot for {packed_key} with index {index} and step {step}")
        
        while self.values[index] is not None and self.keys[index] != packed_key:
            index = (index + step) % self.size
        return index
        
    def _resize(self):
        old_keys, old_values = self.keys, self.values
        self.size = self.size * 2
        self.used = 0
        self.keys = np.zeros(self.size, dtype=np.uint32)
        self.values = np.array([None] * self.size, dtype=object)

        # self.PRIME1 = self.__get_largest_prime(self.size - 1)
        self.PRIME1 = self._next_prime(self.size // 2 - 1)
        
        for key, value in zip(old_keys, old_values):
            if value is not None:
                index = self._find_slot(key)
                self.keys[index] = key
                self.values[index] = value
                self.used += 1

    def memory_usage(self):
        expected_size = (
            self.size * (
                np.dtype(np.uint32).itemsize +  # keys
                np.dtype(object).itemsize       # values
            )
        )
        return {
            'shallow': sum(sys.getsizeof(arr) for arr in [self.keys, self.values]),
            'deep': asizeof.asizeof(self.keys) + asizeof.asizeof(self.values),
            'expected': expected_size
        }

class CustomHashBitpackedDict:
    def __init__(self, initial_size=1024):
        self.size = initial_size
        self.used = 0
        self.keys = [None] * self.size
        self.values = [None] * self.size
    
    def __setitem__(self, key, value):
        packed_key = self._pack_key(*key)
        # index = self._find_slot(packed_key)
        index = self._find_insert_slot(packed_key)
        
        if self.keys[index] is None:
            self.used += 1
            
        self.keys[index] = packed_key
        self.values[index] = value
        
        if self.used > 0.7 * self.size:
            self._resize()

    def __getitem__(self, key):
        packed_key = self._pack_key(*key)
        index = self._find_slot(packed_key)
        
        if self.keys[index] is None:
            raise KeyError(key)
        
        return self.values[index]

    def _pack_key(self, img_id, concept_id):
        if img_id < 0 or img_id >= (1 << 22):
            raise ValueError("img_id must be between 0 and 2^22 - 1")
        if concept_id < 0 or concept_id >= (1 << 10):
            raise ValueError("concept_id must be between 0 and 2^10 - 1")
        return (img_id << 10) | concept_id

    def _find_insert_slot(self, packed_key):
        index = packed_key % self.size
        while self.values[index] is not None:
            index = (index + 1) % self.size
        return index
    
    def _find_slot(self, packed_key):
        index = packed_key % self.size
        while self.keys[index] is not None and self.keys[index] != packed_key:
            index = (index + 1) % self.size
        return index
    
    def _resize(self):
        old_keys, old_values = self.keys, self.values
        self.size *= 2
        self.used = 0
        self.keys = [None] * self.size
        self.values = [None] * self.size
        
        for key, value in zip(old_keys, old_values):
            if key is not None:
                index = self._find_slot(key)
                self.keys[index] = key
                self.values[index] = value
                self.used += 1

    def memory_usage(self):
        return {
            'shallow': sys.getsizeof(self.keys) + sys.getsizeof(self.values),
            'deep': asizeof.asizeof(self.keys) + asizeof.asizeof(self.values)
        }

class DictBasedBitpackedDict:
    def __init__(self):
        self.data = {}
    
    def __setitem__(self, key, value):
        packed_key = self._pack_key(*key)
        self.data[packed_key] = value
    
    def __getitem__(self, key):
        packed_key = self._pack_key(*key)
        return self.data[packed_key]
    
    def _pack_key(self, img_id, concept_id):
        if img_id < 0 or img_id >= (1 << 22):
            raise ValueError("img_id must be between 0 and 2^22 - 1")
        if concept_id < 0 or concept_id >= (1 << 10):
            raise ValueError("concept_id must be between 0 and 2^10 - 1")
        return (img_id << 10) | concept_id
    
    def memory_usage(self):
        return {
            'shallow': sys.getsizeof(self.data),
            'deep': asizeof.asizeof(self.data)
        }

class RegularDict(dict):
    def memory_usage(self):
        return {
            'shallow': sys.getsizeof(self),
            'deep': asizeof.asizeof(self)
        }

def run_single_benchmark(dict_class, n_entries):
    dict_obj = dict_class()
    all_keys = set()
    test_keys = []

    # Insertion benchmark
    start_time = time.time()
    for i in range(n_entries):
        while True:
            img_id = random.randint(0, (1 << 22) - 1)
            concept_id = random.randint(0, (1 << 10) - 1)
            key = (img_id, concept_id)
            if key not in all_keys:
                all_keys.add(key)
                break

        value = f"value_{i}"
        
        if dict_class == RegularDict:
            dict_obj[key] = value
        else:
            dict_obj[img_id, concept_id] = value

        if i % 1000 == 0:
            test_keys.append(key)

    insertion_time = time.time() - start_time

    # Retrieval benchmark and assertions
    start_time = time.time()
    for key in test_keys:
        img_id, concept_id = key
        if dict_class == RegularDict:
            retrieved_value = dict_obj[key]
        else:
            retrieved_value = dict_obj[img_id, concept_id]
        
        # Assert that the retrieved value is correct
        assert retrieved_value == f"value_{test_keys.index(key) * 1000}", f"Incorrect value retrieved for key {key}"

    # Test non-existing keys
    for _ in range(1000):  # Test 100 non-existing keys
        while True:
            img_id = random.randint(0, (1 << 22) - 1)
            concept_id = random.randint(0, (1 << 10) - 1)
            key = (img_id, concept_id)
            if key not in all_keys:
                break
        
        # Assert that KeyError is raised for non-existing keys
        try:
            if dict_class == RegularDict:
                _ = dict_obj[key]
            else:
                _ = dict_obj[img_id, concept_id]
            assert False, f"KeyError not raised for non-existing key {key}"
        except KeyError:
            pass  # This is the expected behavior

    retrieval_time = time.time() - start_time

    # Memory usage
    memory_usage = dict_obj.memory_usage()

    return {
        'insertion_time': insertion_time,
        'retrieval_time': retrieval_time,
        'memory_usage': memory_usage
    }

def run_benchmark(n_entries):
    dict_classes = [NumpyBitpackedDict, CustomHashBitpackedDict, DictBasedBitpackedDict, RegularDict]
    results = {}

    for dict_class in dict_classes:
        class_name = dict_class.__name__
        print(f"\nRunning benchmark for {class_name} with {n_entries} entries")
        try:
            results[class_name] = run_single_benchmark(dict_class, n_entries)
            print(f"All assertions passed for {class_name}")
        except AssertionError as e:
            print(f"Assertion failed for {class_name}: {str(e)}")
            results[class_name] = None

    # Print results
    print(f"\nResults for {n_entries} entries:")
    for class_name, result in results.items():
        if result is not None:
            print(f"\n{class_name}:")
            print(f"Insertion time: {result['insertion_time']:.4f} seconds")
            print(f"Retrieval time: {result['retrieval_time']:.4f} seconds")
            print(f"Memory usage: {result['memory_usage']}")
        else:
            print(f"\n{class_name}: Failed assertions")


if __name__ == '__main__':
    # Run benchmark
    for n in [1000, 10000, 50000, 100000]:
        print(f"\n{'='*40}")
        print(f"Benchmark for {n} entries")
        print(f"{'='*40}")
        run_benchmark(n)