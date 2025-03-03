import bisect

class RangeBasedMapping:
    def __init__(self):
        self.ranges = []  # List of (start_idx, end_idx, img_idx) tuples
        self.img_names = []  # List of image names

    def add_image(self, start_idx, end_idx, img_name):
        # Check for overlaps
        insert_pos = bisect.bisect_left(self.ranges, (start_idx,))
        if insert_pos > 0 and self.ranges[insert_pos-1][1] >= start_idx:
            raise ValueError("New range overlaps with an existing range")
        if insert_pos < len(self.ranges) and end_idx >= self.ranges[insert_pos][0]:
            raise ValueError("New range overlaps with an existing range")

        # Add the new image name
        img_idx = len(self.img_names)
        self.img_names.append(img_name)

        # Insert the new range at the correct position
        bisect.insort(self.ranges, (start_idx, end_idx, img_idx))

    def get_image_name(self, vector_idx):
        img_idx = self._binary_search(vector_idx)
        if img_idx is not None:
            return self.img_names[img_idx]
        return None

    def _binary_search(self, vector_idx):
        left, right = 0, len(self.ranges) - 1

        while left <= right:
            mid = (left + right) // 2
            start, end, img_idx = self.ranges[mid]

            if start <= vector_idx <= end:
                return img_idx
            elif vector_idx < start:
                right = mid - 1
            else:
                left = mid + 1

        return None