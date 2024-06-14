import random


class ProofReturnStatus:

    UNKNOWN = 'unknown'
    TIMEOUT = 'timeout'
    CERTIFIED = 'certified'


class Node:
    
    def __init__(self, history, name='node') -> None:
        self.history = history
        self.name = name
        
    def __len__(self):
        return len(self.history)
        
    def __lt__(self, other):
        "Compare solution spaces"
        if len(self) < len(other):
            return False
        for item in other.history:
            if item not in self.history:
                return False
        return True
    
    def __floordiv__(self, num):
        assert num >= 1
        if not len(self.history):
            return None
        return Node(history=self.history[:int(len(self)/num)], name=f'{self.name}_prefix')
    
    def __repr__(self):
        return f'Node({self.name}, {self.history})'


class ProofQueue:
    
    def __init__(self, proofs: list) -> None:
        histories = proofs if len(proofs) else [[]]
        self.queue = [Node(history=h, name=f'node_{i}') for i, h in enumerate(histories)]
        
    def get(self, batch):
        indices = random.sample(range(len(self)), min(len(self), batch))
        # print(f'{batch=} {len(self)=} {indices=}')
        return [self.queue[idx] for idx in indices]
    
    def add(self, node: Node):
        self.queue.append(node)
    
    def filter(self, node: Node):
        "Filter out solved nodes"
        new_queue = [n for n in self.queue if not n < node]
        self.queue = new_queue
    
    def get_possible_filtered_nodes(self, node):
        if not len(node.history):
            return 1
        nodes = [n for n in self.queue if n < node]
        return len(nodes)
    
    def __len__(self):
        return len(self.queue)

    def __repr__(self):
        lists = []
        if len(self) > 10:
            lists += [str(n) for n in self.queue[:5]]
            lists += ['...']
            lists += [str(n) for n in self.queue[-5:]]
        else:
            lists += [str(n) for n in self.queue]
        return '\nQueue(\n\t' + '\n\t'.join(lists) + '\n)'
            
            
if __name__ == "__main__":
    node_1 = Node(name='aaaa', history=[2])
    node_2 = Node(name='bbbb', history=[1, 2, -4])
    
    print(f'{node_1 < node_2 = }')
    print(f'{node_2 < node_1 = }')
    
    queue = ProofQueue([[-4], [-2, 4], [2, 1, 4], [2, -1, 4]])
    print(queue)
    queue.add(node_1)
    queue.add(node_2)
    print(queue)
    queue.filter(node_1)
    print(queue)
    
    node_3 = node_2 // 2
    print(f'{node_3 = }')
    node_4 = node_3 // 2
    print(f'{node_4 = }')
    node_5 = node_4 // 2
    print(f'{node_5 = }')