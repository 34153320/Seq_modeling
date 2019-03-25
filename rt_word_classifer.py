import numpy as np
import tensorflow as tf

from . import seq_att_classifer as SAC
from RT.util import RTProcesses

from RT.util import fileHandler


class RTSACModel(RTProcesses.RTProcess):
      """Seq-attention model used for word decoding
      """
      def __init__(self, working_dir=None, relevant_elecs=None,
                   name='Seq_att_Model', **kwargs)
          super(RTSACmodel, self).__init__(name=name, **kwargs)
          self.working_dir = working_dir

          self.relevant_elecs = electrodes.obtain_relevant_elecs(relevant_elecs)
          self._session      = None
          self._outputs      = None
          self._input_tensor = None
          self._seq_length   = None

          self.subject       = subject

      def run(self):
          session_config = SAC.get_session_config()
          with tf.Graph().as_default():
               with tf.Session(config=session_config) as self._session:
                     super().run()

      def initializeProcess(self):
          """
            Initializes the decoding model.
          """

          # Displays a loading message
          self.formatMessage('Loading Tandem model...')

        # Restores the tensorflow session and initializes the graph
          self._input_tensor = tf.placeholder(
                tf.float32, [1, None, len(self.relevant_elecs)], name='input_ECoG'
          )
          self._seq_length = tf.placeholder(tf.int32, [1], name='input_len_ECoG')
          self._outputs = td.create_inference_graph(
                                    self._input_tensor, self._seq_length)

          path_pre = tf.train.latest_checkpoint(
               td.params['checkpoint_dir']
          )
          pretrain_saver = tf.train.Saver()
        #  init_op  = tf.global_variables_initializer()
        #  self._session(init_op)
          pretrain_saver.restore(self._session, path_pre)

          self.formatMessage('Successfully loaded Tandem model')


      def main(self):
          """
          """
          i = 0
          while True:
                # Reads a data sample from the input pipe
                input_data = self.readInputPipeWithTimeout()

                # Down-samples the data by a factor of 2
                input_data = input_data[::5, :]

                # Performs the decoding
                feed_dict = {self._input_tensor : [input_data],
                             self._seq_length   : [len(input_data)]}
                decode_ = self._session.run([self._outputs],
                                           feed_dict=feed_dict)

                i += 1
                )
                # Sends the decoded text through the output pipe(s)
                self.sendData(decoded_text)
                self.formatMessage(decoded_text)
