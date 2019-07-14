import tensorflow as tf
from torch_ai import ReplayMemory as RM
from tf_ai import ReplayMemory

tf.enable_eager_execution()


class ReplayMemoryTest(tf.test.TestCase):

    def test_tf_replaymemory(self):
        memory = ReplayMemory(1000)
        for i in range(1000):
            memory.push((i,i,i,i,i))
        batch1, batch2, batch3, batch4, batch5 = memory.sample(10)

        print(batch1)
        print(batch2)
        print(batch3)
        print(batch4)
        print(batch5)

        assert (batch1.shape[0] == batch2.shape[0])

    def test_torch_replaymemory(self):
        memory = RM(1000)
        for i in range(1000):
            memory.push((i,i,i,i,i))
        batch1, batch2, batch3, batch4, batch5 = memory.sample(10)

        print(batch1)
        print(batch2)
        print(batch3)
        print(batch4)
        print(batch5)

        assert (len(batch1) == len(batch2))

if __name__ == '__main__':
      tf.test.main()
