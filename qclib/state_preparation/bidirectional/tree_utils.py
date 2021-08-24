from graphviz    import Digraph

def is_leaf(tree):
    """
    :param tree: a tree node
    :return: True if tree is a leaf
    """
    if tree.left is None and tree.right is None:
        return True

    return False

def remove_leafs(tree):
    if tree.left:
        if is_leaf(tree.left):
            tree.left = None
        else:
            remove_leafs(tree.left)

    if tree.right:
        if is_leaf(tree.right):
            tree.right = None
        else:
            remove_leafs(tree.right)

def leftmost(tree):
    """
    :param tree: a tree node
    :return: the leftmost node relative to tree, or None if tree is leaf.
    """
    if tree.left:
        return tree.left
    else:
        return tree.right

def node_index(tree):
    """
    :param tree: a tree node
    :return: the total index of the node in the tree.
    """
    return 2**tree.level-1+tree.index

def root_node(tree, level):
    """
    :param tree: a tree node
    :param level: level of the subtree (0 for the tree root)
    :return: subtree root at level
    """
    root = tree 
    while root.level > level:
        root = root.parent
        
    return root

def children(nodes):
    """
    Search and list all the nodes childs.
    :param nodes: a list with tree nodes
    :return: a list with nodes childs
    """
    children = []
    for node in nodes:
        if node.left:
            children.append(node.left)
        if node.right:
            children.append(node.right)

    return children

def flatten(tree):
    f = []
    c = [tree]
    while len(c) > 0:
        f.extend(c)
        c = children(c)
    return f

def length(tree):
    """
    Count the total number of the tree nodes.
    :param tree: a tree node
    :return: the total of nodes in the subtree
    """
    if tree:
        n = length(tree.left)
        n += length(tree.right)
        n += 1
        return n 
    return 0

def level_length(tree, level):
    """
    Count the total number of the tree nodes in the level.
    :param tree: a tree node
    :param level: a tree level
    :return: the total of nodes in the subtree level
    """
    if tree:
        if tree.level < level:
            n = level_length(tree.left, level)
            n += level_length(tree.right, level)
            return n
        else:
            return 1
    return 0

def height(root):
    """
    Count the number of levels in the tree.
    :param root: subtree root node
    :return: the total of levels in the subtree defined by root
    """
    n = 0
    left = root
    while left:
        n += 1
        left = leftmost(left)

    return n

def left_view(root, stop_level):
    """
    :param root: subtree root node
    :param stop_level: level below root to stop the search
    :return: list of leftmost nodes between root level and stop_level
    """
    branch = []
    left = root
    while left and left.level <= stop_level:
        branch.append(left)
        left = leftmost(left)
        
    return branch

def subtree_level_index(root, tree):
    """
    :param root: subtree root node
    :param tree: a tree node
    :return: the index of tree node repective to the subtree defined by root
    """
    return tree.index - root.index * 2 ** (tree.level - root.level)

def subtree_level_leftmost(root, level):
    """
    :param root: subtree root node
    :param level: level to search for the leftmost node
    :return: the leftmost tree node repective to the subtree defined by root
    """
    left = root
    while left and left.level < level:
        left = leftmost(left)
    return left

def subtree_level_nodes(tree, level, level_nodes):
    """
    Search and list all the nodes in the indicated level of the tree defined by
    the first value of tree (subtree root).
    :param tree: current tree node, starts with subtree root node
    :param level: level to search for the nodes
    :out param level_nodes: a list with the level tree nodes repective to the 
                            subtree defined by root, ordered from left to right
    """
    if tree.level < level:
        if tree.left:
            subtree_level_nodes(tree.left, level, level_nodes)
        if tree.right:
            subtree_level_nodes(tree.right, level, level_nodes)
    else:
        level_nodes.append(tree)

def tree_visual_representation(tree, dot=None):
    """
    :param tree: A binary tree, with str(tree) defined
    """
    
    if dot is None:
        dot = Digraph()
        dot.node(str(tree))

    if tree.left:
        dot.node(str(tree.left))
        dot.edge(str(tree), str(tree.left))
        dot = tree_visual_representation(tree.left, dot=dot)

    if tree.right:
        dot.node(str(tree.right))
        dot.edge(str(tree), str(tree.right))
        dot = tree_visual_representation(tree.right, dot=dot)

    return dot
