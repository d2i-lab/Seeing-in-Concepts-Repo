import math
from typing import List, Callable, Any, Union

class RBush:
    def __init__(self, max_entries: int = 9):
        # max entries in a node is 9 by default; min node fill is 40% for best performance
        self._max_entries = max(4, max_entries)
        self._min_entries = max(2, math.ceil(self._max_entries * 0.4))
        self.clear()

    def all(self) -> List[Any]:
        return self._all(self.data, [])

    def search(self, bbox: dict) -> List[Any]:
        node = self.data
        result = []

        if not intersects(bbox, node):
            return result

        to_bbox = self.to_bbox
        nodes_to_search = []

        while node:
            for child in node['children']:
                child_bbox = to_bbox(child) if node['leaf'] else child

                if intersects(bbox, child_bbox):
                    if node['leaf']:
                        result.append(child)
                    elif contains(bbox, child_bbox):
                        self._all(child, result)
                    else:
                        nodes_to_search.append(child)

            node = nodes_to_search.pop() if nodes_to_search else None

        return result

    def collides(self, bbox: dict) -> bool:
        node = self.data

        if not intersects(bbox, node):
            return False

        nodes_to_search = []
        while node:
            for child in node['children']:
                child_bbox = self.to_bbox(child) if node['leaf'] else child

                if intersects(bbox, child_bbox):
                    if node['leaf'] or contains(bbox, child_bbox):
                        return True
                    nodes_to_search.append(child)

            node = nodes_to_search.pop() if nodes_to_search else None

        return False

    def load(self, data: List[Any]):
        if not data:
            return self

        if len(data) < self._min_entries:
            for item in data:
                self.insert(item)
            return self

        # recursively build the tree with the given data from scratch using OMT algorithm
        node = self._build(data[:], 0, len(data) - 1, 0)

        if not self.data['children']:
            # save as is if tree is empty
            self.data = node
        elif self.data['height'] == node['height']:
            # split root if trees have the same height
            self._split_root(self.data, node)
        else:
            if self.data['height'] < node['height']:
                # swap trees if inserted one is bigger
                self.data, node = node, self.data

            # insert the small tree into the large tree at appropriate level
            self._insert(node, self.data['height'] - node['height'] - 1, True)

        return self

    def insert(self, item: Any):
        if item:
            self._insert(item, self.data['height'] - 1)
        return self

    def clear(self):
        self.data = create_node([])
        return self

    def remove(self, item: Any, equals_fn: Callable[[Any, Any], bool] = None):
        if not item:
            return self

        node = self.data
        bbox = self.to_bbox(item)
        path = []
        indexes = []
        parent = None
        index = 0
        going_up = False

        # depth-first iterative tree traversal
        while node or path:
            if not node:  # go up
                node = path.pop()
                parent = path[-1] if path else None
                index = indexes.pop()
                going_up = True

            if node['leaf']:  # check current node
                idx = find_item(item, node['children'], equals_fn)

                if idx != -1:
                    # item found, remove the item and condense tree upwards
                    node['children'].pop(idx)
                    path.append(node)
                    self._condense(path)
                    return self

            if not going_up and not node['leaf'] and contains(node, bbox):  # go down
                path.append(node)
                indexes.append(index)
                index = 0
                parent = node
                node = node['children'][0]
            elif parent:  # go right
                index += 1
                node = parent['children'][index] if index < len(parent['children']) else None
                going_up = False
            else:
                node = None  # nothing found

        return self

    def to_bbox(self, item: Any) -> dict:
        return item

    def compare_min_x(self, a: dict, b: dict) -> float:
        return a['minX'] - b['minX']

    def compare_min_y(self, a: dict, b: dict) -> float:
        return a['minY'] - b['minY']

    def to_json(self) -> dict:
        return self.data

    def from_json(self, data: dict):
        self.data = data
        return self

    def _all(self, node: dict, result: List[Any]) -> List[Any]:
        nodes_to_search = []
        while node:
            if node['leaf']:
                result.extend(node['children'])
            else:
                nodes_to_search.extend(node['children'])

            node = nodes_to_search.pop() if nodes_to_search else None
        return result
    
    def _build(self, items: List[Any], left: int, right: int, height: int) -> dict:
        N = right - left + 1
        M = self._max_entries

        if N <= M:
            # reached leaf level; return leaf
            node = create_node(items[left:right + 1])
            calc_bbox(node, self.to_bbox)
            return node

        if not height:
            # target height of the bulk-loaded tree
            height = math.ceil(math.log(N) / math.log(M))

            # target number of root entries to maximize storage utilization
            M = math.ceil(N / (M ** (height - 1)))

        node = create_node([])
        node['leaf'] = False
        node['height'] = height

        # split the items into M mostly square tiles

        N2 = math.ceil(N / M)
        N1 = N2 * math.ceil(math.sqrt(M))

        multi_select(items, left, right, N1, self.compare_min_x)

        for i in range(left, right + 1, N1):
            right2 = min(i + N1 - 1, right)

            multi_select(items, i, right2, N2, self.compare_min_y)

            for j in range(i, right2 + 1, N2):
                right3 = min(j + N2 - 1, right2)

                # pack each entry recursively
                node['children'].append(self._build(items, j, right3, height - 1))

        calc_bbox(node, self.to_bbox)

        return node

    def _choose_subtree(self, bbox: dict, node: dict, level: int, path: List[dict]) -> dict:
        while True:
            path.append(node)
            if node['leaf'] or len(path) - 1 == level:
                break

            min_area = min_enlargement = float('inf')
            target_node = None

            for child in node['children']:
                area = bbox_area(child)
                enlargement = enlarged_area(bbox, child) - area

                # choose entry with the least area enlargement
                if enlargement < min_enlargement:
                    min_enlargement = enlargement
                    min_area = min(area, min_area)
                    target_node = child
                elif enlargement == min_enlargement:
                    # otherwise choose one with the smallest area
                    if area < min_area:
                        min_area = area
                        target_node = child

            node = target_node or node['children'][0]

        return node

    def _insert(self, item: Any, level: int, is_node: bool = False):
        bbox = item if is_node else self.to_bbox(item)
        insert_path = []

        # find the best node for accommodating the item, saving all nodes along the path too
        node = self._choose_subtree(bbox, self.data, level, insert_path)

        # put the item into the node
        node['children'].append(item)
        extend(node, bbox)

        # split on node overflow; propagate upwards if necessary
        while level >= 0:
            if len(insert_path[level]['children']) > self._max_entries:
                self._split(insert_path, level)
                level -= 1
            else:
                break

        # adjust bboxes along the insertion path
        self._adjust_parent_bboxes(bbox, insert_path, level)

    def _split(self, insert_path: List[dict], level: int):
        node = insert_path[level]
        M = len(node['children'])
        m = self._min_entries

        self._choose_split_axis(node, m, M)

        split_index = self._choose_split_index(node, m, M)

        new_node = create_node(node['children'][split_index:])
        node['children'] = node['children'][:split_index]
        new_node['height'] = node['height']
        new_node['leaf'] = node['leaf']

        calc_bbox(node, self.to_bbox)
        calc_bbox(new_node, self.to_bbox)

        if level:
            insert_path[level - 1]['children'].append(new_node)
        else:
            self._split_root(node, new_node)

    def _split_root(self, node: dict, new_node: dict):
        # split root node
        self.data = create_node([node, new_node])
        self.data['height'] = node['height'] + 1
        self.data['leaf'] = False
        calc_bbox(self.data, self.to_bbox)

    def _choose_split_index(self, node: dict, m: int, M: int) -> int:
        min_overlap = min_area = float('inf')
        index = m

        for i in range(m, M - m + 1):
            bbox1 = dist_bbox(node, 0, i, self.to_bbox)
            bbox2 = dist_bbox(node, i, M, self.to_bbox)

            overlap = intersection_area(bbox1, bbox2)
            area = bbox_area(bbox1) + bbox_area(bbox2)

            # choose distribution with minimum overlap
            if overlap < min_overlap:
                min_overlap = overlap
                index = i

                min_area = min(area, min_area)

            elif overlap == min_overlap:
                # otherwise choose distribution with minimum area
                if area < min_area:
                    min_area = area
                    index = i

        return index

    def _choose_split_axis(self, node: dict, m: int, M: int):
        compare_min_x = self.compare_min_x if node['leaf'] else compare_node_min_x
        compare_min_y = self.compare_min_y if node['leaf'] else compare_node_min_y

        x_margin = self._all_dist_margin(node, m, M, compare_min_x)
        y_margin = self._all_dist_margin(node, m, M, compare_min_y)

        # if total distributions margin value is minimal for x, sort by minX,
        # otherwise it's already sorted by minY
        if x_margin < y_margin:
            node['children'].sort(key=lambda child: child['minX'])

    def _all_dist_margin(self, node: dict, m: int, M: int, compare_fn: Callable[[Any, Any], float]) -> float:
        node['children'].sort(key=lambda child: compare_fn(child, child))

        left_bbox = dist_bbox(node, 0, m, self.to_bbox)
        right_bbox = dist_bbox(node, M - m, M, self.to_bbox)
        margin = bbox_margin(left_bbox) + bbox_margin(right_bbox)

        for i in range(m, M - m):
            child = node['children'][i]
            extend(left_bbox, node['leaf'] and self.to_bbox(child) or child)
            margin += bbox_margin(left_bbox)

        for i in range(M - m - 1, m - 1, -1):
            child = node['children'][i]
            extend(right_bbox, node['leaf'] and self.to_bbox(child) or child)
            margin += bbox_margin(right_bbox)

        return margin

    def _adjust_parent_bboxes(self, bbox: dict, path: List[dict], level: int):
        # adjust bboxes along the given tree path
        for i in range(level, -1, -1):
            extend(path[i], bbox)

    def _condense(self, path: List[dict]):
        # go through the path, removing empty nodes and updating bboxes
        for i in range(len(path) - 1, -1, -1):
            if not path[i]['children']:
                if i > 0:
                    siblings = path[i - 1]['children']
                    siblings.remove(path[i])
                else:
                    self.clear()
            else:
                calc_bbox(path[i], self.to_bbox)


# Helper functions

def find_item(item: Any, items: List[Any], equals_fn: Callable[[Any, Any], bool] = None) -> int:
    if not equals_fn:
        return items.index(item) if item in items else -1

    for i, it in enumerate(items):
        if equals_fn(item, it):
            return i
    return -1

def create_node(children: List[Any]) -> dict:
    return {
        'children': children,
        'height': 1,
        'leaf': True,
        'minX': float('inf'),
        'minY': float('inf'),
        'maxX': float('-inf'),
        'maxY': float('-inf')
    }

def extend(a: dict, b: dict) -> dict:
    a['minX'] = min(a['minX'], b['minX'])
    a['minY'] = min(a['minY'], b['minY'])
    a['maxX'] = max(a['maxX'], b['maxX'])
    a['maxY'] = max(a['maxY'], b['maxY'])
    return a

def contains(a: dict, b: dict) -> bool:
    return (
        a['minX'] <= b['minX'] and
        a['minY'] <= b['minY'] and
        b['maxX'] <= a['maxX'] and
        b['maxY'] <= a['maxY']
    )

def intersects(a: dict, b: dict) -> bool:
    return (
        b['minX'] <= a['maxX'] and
        b['minY'] <= a['maxY'] and
        b['maxX'] >= a['minX'] and
        b['maxY'] >= a['minY']
    )


# Helper functions

def calc_bbox(node: dict, to_bbox: Callable[[Any], dict]):
    dist_bbox(node, 0, len(node['children']), to_bbox, node)

def dist_bbox(node: dict, k: int, p: int, to_bbox: Callable[[Any], dict], dest_node: dict = None) -> dict:
    if dest_node is None:
        dest_node = create_node(None)

    for i in range(k, p):
        child = node['children'][i]
        extend(dest_node, node['leaf'] and to_bbox(child) or child)

    return dest_node

def compare_node_min_x(a: dict, b: dict) -> float:
    return a['minX'] - b['minX']

def compare_node_min_y(a: dict, b: dict) -> float:
    return a['minY'] - b['minY']

def bbox_area(a: dict) -> float:
    return (a['maxX'] - a['minX']) * (a['maxY'] - a['minY'])

def bbox_margin(a: dict) -> float:
    return (a['maxX'] - a['minX']) + (a['maxY'] - a['minY'])

def enlarged_area(a: dict, b: dict) -> float:
    return ((max(b['maxX'], a['maxX']) - min(b['minX'], a['minX'])) *
            (max(b['maxY'], a['maxY']) - min(b['minY'], a['minY'])))

def intersection_area(a: dict, b: dict) -> float:
    minX = max(a['minX'], b['minX'])
    minY = max(a['minY'], b['minY'])
    maxX = min(a['maxX'], b['maxX'])
    maxY = min(a['maxY'], b['maxY'])

    return max(0, maxX - minX) * max(0, maxY - minY)

def multi_select(arr: List[Any], left: int, right: int, n: int, compare: Callable[[Any, Any], float]):
    stack = [left, right]

    while stack:
        right = stack.pop()
        left = stack.pop()

        if right - left <= n:
            continue

        mid = left + math.ceil((right - left) / n / 2) * n
        quick_select(arr, mid, left, right, compare)

        stack.extend([left, mid, mid, right])

def quick_select(arr: List[Any], k: int, left: int, right: int, compare: Callable[[Any, Any], float]):
    while right > left:
        if right - left > 600:
            n = right - left + 1
            m = k - left + 1
            z = math.log(n)
            s = 0.5 * math.exp(2 * z / 3)
            sd = 0.5 * math.sqrt(z * s * (n - s) / n) * (m - n / 2 < 0 and -1 or 1)
            new_left = max(left, math.floor(k - m * s / n + sd))
            new_right = min(right, math.floor(k + (n - m) * s / n + sd))
            quick_select(arr, k, new_left, new_right, compare)

        t = arr[k]
        i = left
        j = right

        arr[left], arr[k] = arr[k], arr[left]
        if compare(arr[right], t) > 0:
            arr[right], arr[left] = arr[left], arr[right]

        while i < j:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j -= 1
            while compare(arr[i], t) < 0:
                i += 1
            while compare(arr[j], t) > 0:
                j -= 1

        if compare(arr[left], t) == 0:
            arr[left], arr[j] = arr[j], arr[left]
        else:
            j += 1
            arr[j], arr[right] = arr[right], arr[j]

        if j <= k:
            left = j + 1
        if k <= j:
            right = j - 1