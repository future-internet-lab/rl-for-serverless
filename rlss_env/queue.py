class QueueEmptyError(Exception):
    pass

class QueueFullError(Exception):    
    pass

class CircularQueue:
    def __init__(self, size):
        self.size = size
        self.queue = [None] * size
        self.front = -1
        self.rear = -1

    def is_empty(self):
        return self.front == -1

    def is_full(self):
        return (self.rear + 1) % self.size == self.front

    def enqueue(self, item):
        if self.is_full():
            self.front = (self.front + 1) % self.size
        else:
            if self.front == -1:
                self.front = 0
            self.rear = (self.rear + 1) % self.size
        
        self.queue[self.rear] = item

    def dequeue(self):
        if self.is_empty():
            raise QueueEmptyError("Queue is empty. Cannot dequeue.")
        else:
            item = self.queue[self.front]
            if self.front == self.rear: 
                self.front = self.rear = -1
            else:
                self.front = (self.front + 1) % self.size
            return item

    def peek(self):
        if self.is_empty():
            raise QueueEmptyError("Queue is empty.")
        else:
            return self.queue[self.front]

    def display(self):
        if self.is_empty():
            raise QueueEmptyError("Queue is empty.")
        else:
            result = []
            i = self.front
            while True:
                result.append(self.queue[i])
                if i == self.rear:
                    break
                i = (i + 1) % self.size
            return result

    def size(self):
        if self.is_empty():
            return 0
        elif self.rear >= self.front:
            return self.rear - self.front + 1
        else:
            return self.size - (self.front - self.rear - 1)

