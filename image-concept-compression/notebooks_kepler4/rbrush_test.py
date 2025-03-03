from rbrush import RBush
importlib.reload(rbrush)

def test_insert_and_search():
    rbush = RBush(max_entries=4)
    items = [
        {'minX': 0, 'minY': 0, 'maxX': 1, 'maxY': 1, 'data': 'A'},
        {'minX': 1, 'minY': 1, 'maxX': 2, 'maxY': 2, 'data': 'B'},
        {'minX': 2, 'minY': 2, 'maxX': 3, 'maxY': 3, 'data': 'C'},
    ]
    for item in items:
        rbush.insert(item)

    results = rbush.search({'minX': 0, 'minY': 0, 'maxX': 3, 'maxY': 3})
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all(item in results for item in items), "Not all inserted items were found in search results"
    print("test_insert_and_search: PASSED")

def test_search_empty_tree():
    rbush = RBush(max_entries=4)
    results = rbush.search({'minX': 0, 'minY': 0, 'maxX': 1, 'maxY': 1})
    assert len(results) == 0, f"Expected 0 results, got {len(results)}"
    print("test_search_empty_tree: PASSED")

def test_search_no_results():
    rbush = RBush(max_entries=4)
    rbush.insert({'minX': 0, 'minY': 0, 'maxX': 1, 'maxY': 1, 'data': 'A'})
    results = rbush.search({'minX': 2, 'minY': 2, 'maxX': 3, 'maxY': 3})
    assert len(results) == 0, f"Expected 0 results, got {len(results)}"
    print("test_search_no_results: PASSED")

def test_insert_many_items():
    rbush = RBush(max_entries=4)
    items = [{'minX': i, 'minY': i, 'maxX': i+1, 'maxY': i+1, 'data': str(i)} for i in range(100)]
    for item in items:
        rbush.insert(item)

    results = rbush.search({'minX': 0, 'minY': 0, 'maxX': 100, 'maxY': 100})
    assert len(results) == 100, f"Expected 100 results, got {len(results)}"
    print("test_insert_many_items: PASSED")

def test_clear():
    rbush = RBush(max_entries=4)
    rbush.insert({'minX': 0, 'minY': 0, 'maxX': 1, 'maxY': 1, 'data': 'A'})
    rbush.clear()
    results = rbush.search({'minX': 0, 'minY': 0, 'maxX': 1, 'maxY': 1})
    assert len(results) == 0, f"Expected 0 results after clear, got {len(results)}"
    print("test_clear: PASSED")

def test_to_json_and_from_json():
    rbush = RBush(max_entries=4)
    items = [
        {'minX': 0, 'minY': 0, 'maxX': 1, 'maxY': 1, 'data': 'A'},
        {'minX': 1, 'minY': 1, 'maxX': 2, 'maxY': 2, 'data': 'B'},
    ]
    for item in items:
        rbush.insert(item)

    json_data = rbush.to_json()
    new_rbush = RBush(max_entries=4)
    new_rbush.from_json(json_data)

    results = new_rbush.search({'minX': 0, 'minY': 0, 'maxX': 2, 'maxY': 2})
    assert len(results) == 2, f"Expected 2 results after JSON roundtrip, got {len(results)}"
    print("test_to_json_and_from_json: PASSED")

def test_edge_case_single_point():
    rbush = RBush(max_entries=4)
    rbush.insert({'minX': 1, 'minY': 1, 'maxX': 1, 'maxY': 1, 'data': 'Point'})
    results = rbush.search({'minX': 1, 'minY': 1, 'maxX': 1, 'maxY': 1})
    assert len(results) == 1, f"Expected 1 result for single point, got {len(results)}"
    assert results[0]['data'] == 'Point', f"Expected 'Point' data, got {results[0]['data']}"
    print("test_edge_case_single_point: PASSED")

def test_edge_case_overlapping_boxes():
    rbush = RBush(max_entries=4)
    rbush.insert({'minX': 0, 'minY': 0, 'maxX': 2, 'maxY': 2, 'data': 'A'})
    rbush.insert({'minX': 1, 'minY': 1, 'maxX': 3, 'maxY': 3, 'data': 'B'})
    results = rbush.search({'minX': 1, 'minY': 1, 'maxX': 2, 'maxY': 2})
    assert len(results) == 2, f"Expected 2 results for overlapping boxes, got {len(results)}"
    print("test_edge_case_overlapping_boxes: PASSED")

def run_all_tests():
    tests = [
        test_insert_and_search,
        test_search_empty_tree,
        test_search_no_results,
        test_insert_many_items,
        test_clear,
        test_to_json_and_from_json,
        test_edge_case_single_point,
        test_edge_case_overlapping_boxes
    ]

    total_tests = len(tests)
    passed_tests = 0

    for test in tests:
        try:
            test()
            passed_tests += 1
        except AssertionError as e:
            print(f"{test.__name__}: FAILED")
            print(f"  Error: {str(e)}")

    print(f"\nTest Results: {passed_tests}/{total_tests} tests passed.")

# Run all tests
run_all_tests()

# import importlib
# from rbrush import RBush
# importlib.reload(rbrush)

# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# # from rtree_implementation import RBush  # Assuming we saved our implementation in rtree_implementation.py
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

# def visualize_rtree(rtree, ax=None, show_labels=False):
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(12, 8))
    
#     def draw_node(node, level=0):
#         color = plt.cm.viridis(level / 5)  # Color based on tree level
#         rect = Rectangle((node['minX'], node['minY']), 
#                          node['maxX'] - node['minX'], 
#                          node['maxY'] - node['minY'],
#                          fill=False, color=color, linewidth=2)
#         ax.add_patch(rect)
        
#         if node['leaf'] and show_labels:
#             for item in node['children']:
#                 ax.text(item['minX'], item['minY'], item['data'], fontsize=8, 
#                         ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
#         if not node['leaf']:
#             for child in node['children']:
#                 draw_node(child, level + 1)
    
#     draw_node(rtree.data)
#     ax.set_xlim(0, 100)
#     ax.set_ylim(0, 100)
#     ax.set_aspect('equal', 'box')
#     ax.set_title('City Map with Points of Interest')
#     ax.set_xlabel('X coordinate')
#     ax.set_ylabel('Y coordinate')

# # Create some test data (points of interest in a city)
# test_data = [
#     {'minX': 10, 'minY': 10, 'maxX': 10, 'maxY': 10, 'data': 'City Hall'},
#     {'minX': 30, 'minY': 30, 'maxX': 30, 'maxY': 30, 'data': 'Central Park'},
#     {'minX': 50, 'minY': 50, 'maxX': 50, 'maxY': 50, 'data': 'Main Station'},
#     {'minX': 70, 'minY': 70, 'maxX': 70, 'maxY': 70, 'data': 'Shopping Mall'},
#     {'minX': 40, 'minY': 80, 'maxX': 40, 'maxY': 80, 'data': 'University'},
#     {'minX': 20, 'minY': 60, 'maxX': 20, 'maxY': 60, 'data': 'Museum'},
#     {'minX': 60, 'minY': 20, 'maxX': 60, 'maxY': 20, 'data': 'Sports Stadium'},
#     {'minX': 80, 'minY': 40, 'maxX': 80, 'maxY': 40, 'data': 'Hospital'},
#     {'minX': 90, 'minY': 90, 'maxX': 90, 'maxY': 90, 'data': 'Airport'},
# ]

# # Create and populate the R-tree
# rtree = RBush(max_entries=4)
# for item in test_data:
#     rtree.insert(item)

# # Visualize the R-tree structure
# plt.figure(figsize=(12, 8))
# visualize_rtree(rtree, show_labels=True)
# plt.savefig('rtree_structure.png', dpi=300, bbox_inches='tight')
# plt.close()

# # Example of search visualization
# search_bbox = {'minX': 20, 'minY': 20, 'maxX': 90, 'maxY': 60}
# search_results = rtree.search(search_bbox)

# fig, ax = plt.subplots(figsize=(12, 8))
# visualize_rtree(rtree, ax, show_labels=True)

# # Draw search bbox
# search_rect = Rectangle((search_bbox['minX'], search_bbox['minY']),
#                         search_bbox['maxX'] - search_bbox['minX'],
#                         search_bbox['maxY'] - search_bbox['minY'],
#                         fill=False, color='r', linestyle='--', linewidth=2)
# ax.add_patch(search_rect)

# # Highlight search results
# for item in search_results:
#     result_rect = Rectangle((item['minX']-2, item['minY']-2),
#                             4, 4,
#                             fill=True, alpha=0.5, color='r')
#     ax.add_patch(result_rect)

# ax.set_title('City Map: Searching for Points of Interest')
# plt.show()
# # plt.savefig('rtree_search.png', dpi=300, bbox_inches='tight')
# # plt.close()

# print("Visualization complete. Check 'rtree_structure.png' and 'rtree_search.png' for results.")