class ReplayBuffer:
      def __init__(self, buffer_size):
            self.buffer_size = buffer_size
            self.count = 0
            self.buffer = deque()
            
            
      def add(self, cs, a, r, d, ns):
            """Adds an experience to the buffer"""
            # cs: current state
            # a: action
            # r: reward
            # d: done
            # ns: next state
            experience = (cs, a, r, d, ns)
            if self.count < self.buffer_size:
                  self.buffer.append(experience)
                  self.count += 1
            else:
                  self.buffer.popleft()
                  self.buffer.append(experience)
      
      
      def size(self):
            return self.count
      
      
      def sample(self, batch_size):
            """Samples batch_size samples from the buffer, if the
            buffer contains enough elements. Otherwise, returns all elements"""
            
            batch = []
            if self.count < batch_size:
                  batch = random.sample(self.buffer, self.count)
            else:
                  batch = random.sample(self.buffer, batch_size)
            
            # Maps each experience in the batch in batches of current sates,
            # actions, rewards, dones and next states
            cs_batch, a_batch, r_batch, d_batch, ns_batch = list(map(np.array,
                                                                     list(zip(*batch))))
            return cs_batch, a_batch, r_batch, d_batch, ns_batch
      
      
      def clear(self):
            self.buffer.clear()
            self.count = 0    
