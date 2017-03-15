# -*- coding:utf8 -*-
'''
利用Encoder-Decoder写诗的尝试。
'''

import tensorflow as tf
import collections
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import codecs
import numpy as np
from random import sample

def device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"

with tf.Session() as sess:

    with tf.device(device_for_node):

        def batch_op(inputs, max_sequence_length=None):
            '''
            将所有输入填充为max_sequence_length的长度,并返回一个形状为[max_time, batch_size]的输入矩阵
            :param inputs: list of sentences(integer lists)
            :param max_sequence_length: integer specifying how large should 'max_time' dimension be.
                                        If None, maximum sequence length would be used.
            :return:    1. inputs_time_major: input sentences transformed into time-major matrix
                                                (shape [max_time, batch_size]) padded with 0s
                        2. sequence_lengths: batch-sized list of integer specifying amount of active
                                                time steps in each input sequence
            '''

            sequence_lengths = [len(seq) for seq in inputs]
            batch_size = len(inputs)

            if max_sequence_length is None:
                max_sequence_length = max(sequence_lengths)

            inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)

            for i, seq in enumerate(inputs):
                for j, element in enumerate(seq):
                    inputs_batch_major[i,j] = element

            inputs_time_major = inputs_batch_major.swapaxes(0,1)

            return inputs_time_major, sequence_lengths

        # 读取诗词生成输入文本

        poems_file = "../data/poetry.txt"

        poems = []
        with codecs.open(poems_file, 'r', 'utf-8') as f:
            for line in f:
                #print line
                try:
                    title, content = line.strip().split(":")
                    content = content.replace(' ','')
                    #content = content.replace(u'。', '')
                    #content = content.replace(u'，', '')
                    if '_' in content or \
                        '(' in content or \
                        u'《' in content or \
                        '[' in content:
                        continue
                    if len(content) < 5 or len(content) > 79:
                        continue
                    #print content
                    poems.append(content)
                except Exception as e:
                    pass
                    #print e

        poems = sorted(poems, key=lambda line: len(line))
        print "Total poems:", len(poems)

        #poems = poems[:5000]

        all_words = []
        for poem in poems:
            all_words += [word for word in poem]
        counter = collections.Counter(all_words)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])

        words,_ = zip(*count_pairs)

        print "Total words:", len(words)

        #for w in words[:40]:
        #    print w

        PAD = 0
        EOS = 1

        dict = {"<PAD>": 0, "<EOS>": 1}
        reversed_dict = {0: "<PAD>", 1: "<EOS>"}
        for word in words:
            dict[word] = len(dict)
            reversed_dict[len(reversed_dict)] = word

        embedded_poems = []
        for poem in poems:
            embedded_poems.append([dict[word] for word in poem])

        #print embedded_poems

        vocab_size = len(dict)
        word_embedding_size = 128

        encoder_hidden_units = 128
        decoder_hidden_units = encoder_hidden_units * 2

        encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

        decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

        embeddings = tf.Variable(tf.random_uniform([vocab_size, word_embedding_size], -1.0, 1.0), dtype=tf.float32)
        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

        encoder_cell = LSTMCell(encoder_hidden_units)

        ((encoder_fw_outputs,
          encoder_bw_outputs),
         (encoder_fw_final_state,
          encoder_bw_final_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                            cell_bw=encoder_cell,
                                            inputs=encoder_inputs_embedded,
                                            sequence_length=encoder_inputs_length,
                                            dtype=tf.float32,
                                            time_major=True)
        )

        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), axis=2)

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )

        decoder_cell = LSTMCell(decoder_hidden_units)
        encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
        decoder_lengths = encoder_inputs_length + 3

        W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1.0, 1.0), dtype=tf.float32)
        b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

        assert EOS == 1 and PAD == 0

        eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
        pad_time_slice = tf.ones([batch_size], dtype=tf.int32, name='PAD')

        eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
        pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

        def loop_fn_initial():
            initial_elements_finished = ( 0 >= decoder_lengths)
            initial_input = eos_step_embedded
            initial_cell_state = encoder_final_state
            initial_cell_output = None
            initial_loop_state = None
            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state
                    )

        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

            def get_next_input():
                output_logits = tf.add(tf.matmul(previous_output, W), b)
                prediction = tf.argmax(output_logits, axis=1)
                next_input = tf.nn.embedding_lookup(embeddings, prediction)
                return next_input

            elements_finished = (time >= decoder_lengths)

            finished = tf.reduce_all(elements_finished)
            input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
            state = previous_state
            output = previous_output
            loop_state = None

            return (elements_finished,
                    input,
                    state,
                    output,
                    loop_state)

        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            if previous_state is None:
                assert previous_output is None and previous_state is None
                return loop_fn_initial()
            else:
                return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
        decoder_outputs = decoder_outputs_ta.stack()

        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
        decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

        decoder_prediction = tf.argmax(decoder_logits, 2)

        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels = tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
            logits=decoder_logits,
        )

        loss = tf.reduce_mean(stepwise_cross_entropy)
        train_op = tf.train.AdamOptimizer().minimize(loss)

        sess.run(tf.global_variables_initializer())

        '''
        batch_size = 100

        batches = random_sequences(length_from=3, length_to=8,
                                          vocab_lower=2, vocab_upper=10,
                                          batch_size=batch_size)

        def next_feed():
            batch = next(batches)
            encoder_inputs_, encoder_inputs_length_ = batch_op(batch)
            decoder_targets_, _ = batch_op(
                [(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
            )

            return {
                encoder_inputs: encoder_inputs_,
                encoder_inputs_length: encoder_inputs_length_,
                decoder_targets: decoder_targets_,
            }

        loss_track = []

        max_batches = 3001
        batches_in_epoch = 1000

        try:
            for batch in range(max_batches):
                fd = next_feed()
                _, l = sess.run([train_op, loss], fd)
                loss_track.append(l)

                if batch == 0 or batch % batches_in_epoch == 0:
                    print('batch {}'.format(batch))
                    print('  minibatch loss: {}').format(sess.run(loss, fd))
                    predict_ = sess.run(decoder_prediction, fd)
                    for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                        print('  sample {}:'.format(i+1))
                        print('    input     > {}'.format(inp))
                        print('    predicted > {}'.format(pred))
                        if i >= 2:
                            break
                        print ""
        except KeyboardInterrupt:
            print('training interrupted')

        '''
        def next_feed():
            #np.random.shuffle(embedded_poems)
            #print embedded_poems[:5]
            batch_poems = sample(embedded_poems, 100)
            encoder_inputs_, encoder_inputs_length_ = batch_op(batch_poems)
            decoder_targets_, _ = batch_op([(seq) + [EOS] + [PAD] * 2 for seq in batch_poems])

            return {
                encoder_inputs: encoder_inputs_,
                encoder_inputs_length: encoder_inputs_length_,
                decoder_targets: decoder_targets_
            }

        loss_track = []

        max_batches = 10001
        batches_in_epoch = 100

        try:
            for batch in range(max_batches):
                fd = next_feed()
                _, l = sess.run([train_op, loss], fd)
                loss_track.append(l)

                if batch == 0 or batch % batches_in_epoch == 0:
                    print('batch {}'.format(batch))
                    print('  minibatch loss: {}').format(sess.run(loss, fd))
                    predict_ = sess.run(decoder_prediction, fd)
                    for i, (inp, targ, pred) in enumerate(zip(fd[encoder_inputs].T, fd[decoder_targets].T, predict_.T)):
                        print('  sample {}:'.format(i + 1))
                        inpstr = ''
                        for j in inp:
                            inpstr += reversed_dict[j]
                        predstr = ''
                        for j in pred:
                            predstr += reversed_dict[j]
                        tagstr = ''
                        for j in targ:
                            tagstr += reversed_dict[j]
                        print('    input     > ')
                        print inpstr
                        print('    target    > ')
                        print tagstr
                        print('    predicted > ')
                        print predstr
                        if i >= 2:
                            break
                        print ""
        except KeyboardInterrupt:
            print('training interrupted')
