class Queue:
    
    def __init__(self, items=None):
        if items is None:
            items = []
        self.items = items

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            print("Queue is empty")

    def size(self):
        return len(self.items)

    def peek(self):
        if not self.is_empty():
            return self.items[0]
        else:
            print("Queue is empty")


# Example usage:
# q = Queue()
# q.enqueue(1)
# q.enqueue(2)
# q.enqueue(3)

# print("Queue size:", q.size())
# print("Dequeued item:", q.dequeue())
# print("Queue size after dequeue:", q.size())
# print("Peek item:", q.peek())
