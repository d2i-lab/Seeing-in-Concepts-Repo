import re

def query_by_list(df, query_list:list):
    """
    Return the dataframe filtered by query list. The query list is a list of
    columns that represents a slice. Union columns are represented with a |.

    Example: query_list = ['A', 'B', 'C|D'] will is the equivalent of
    A^B^(C|D).

    TODO: Support concept absence. e.g. A=0. Currently assumes all slices are
    for concept presence.
    """
    if type(query_list) == str:
        query_list = [query_list]

    df = df.copy()
    normal_cols = {}
    union_cols = {}
    df_col_set = set(list(df.columns))
    # print(df_col_set)
    for col in query_list:
        # Skip empty strings
        if len(col) == 0:
            continue

        if col in df_col_set:
            col = col
            val = 1
            normal_cols[val] = normal_cols.get(val, []) + [col]
            continue

        val = 1
        if "=" in col:
            col, val = col.split("=")
            val = int(val)


        if "|" in col:
            union_cols[val] = union_cols.get(val, []) + [col]
        else:
            normal_cols[val] = normal_cols.get(val, []) + [col]

    
    if len(normal_cols) > 0:
        for val, items in normal_cols.items():
            df = df[(df[items].astype(int)==val).all(axis=1)]

    if len(union_cols) > 0:
        for val, items in union_cols.items():
            for u in items:
                unions = list(u.split("|"))
                df = df[(df[unions]==val).any(axis=1)]

    return df

def itemset_to_str(itemset:list):
    if len(itemset) == 0:
        return '()'

    slicelist = []
    for item in itemset:
        if '=' not in item:
            slicelist.append('{}=1'.format(item))
        else:
            slicelist.append(item)

    slice_str = '({})'.format(','.join(slicelist))
    return slice_str


def custom_split(input_string):
    # Define a regular expression pattern to match a comma outside of parentheses, square brackets, and mixed brackets.
    pattern = r',(?![^[\]()]*[\])])'
    
    # Use re.split to split the string based on the pattern.
    result = re.split(pattern, input_string)
    
    return result

def parse_slice(itemset):
    if type(itemset) == list:
        itemset = itemset_to_str(itemset)
    itemset = custom_split(itemset.strip('(').strip(')'))

    return itemset

def get_slice(df, itemset):
    '''
    Retrieve rows from dataframe given itemset string
    e.g. get_slice(df, '(0=1, 1=1, 2=1)'
    Also works for list of items:
    e.g. get_slice(df, [0, 1, 2])
    '''
    if type(itemset) == list:
        itemset = itemset_to_str(itemset)
    itemset = custom_split(itemset.strip('(').strip(')'))
    if len(itemset) == 1 and itemset[0] == '':
        return df

    return query_by_list(df, itemset)

def calculate_overlap(df, gt_slice, test_slice, mode=None):
    """
    Given dataframe and a ground truth slice, calculate the overlap between
    slice row queries. If mode is None, calculate the raw overlap.
    If mode is "accuracy", calculate the accuracy of the slice.
    Finally, if the mode is "IOU," calculate the IOU of the slice.
    """
    queried_gt = set(get_slice(df, gt_slice).index)
    queried_test = set(get_slice(df, test_slice).index)
    row_intersection = queried_gt.intersection(queried_test)

    if mode is "intersect":
        return len(row_intersection)
    if mode == "accuracy":
        return len(row_intersection) / len(queried_gt)
    elif mode == "IOU":
        return len(row_intersection) / len(queried_gt.union(queried_test))
    
    raise Exception("Invalid mode {}".format(mode))

