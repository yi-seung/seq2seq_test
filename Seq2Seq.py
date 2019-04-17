# 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq 을 구현해봅니다.
# 영어 단어를 한국어 단어로 번역하는 프로그램을 만들어봅니다.
import tensorflow as tf
import numpy as np

with open("NIV.txt", "r", encoding="utf-8") as f_en:
    lines_en = f_en.readlines()
with open("개역개정4판.txt", "r", encoding="utf-8") as f_ko:
    lines_ko = f_ko.readlines()

seq_data = []
temp_data = [" ", " "]

for i in range(200):
    temp_data = ["{0:<512}".format(lines_en[i]), "{0:<256}".format(lines_ko[i])]
    seq_data.append(temp_data)

print("=================================")
print(seq_data[0:2])


######################################################################
# S: 디코딩 입력의 시작을 나타내는 심볼
# E: 디코딩 출력을 끝을 나타내는 심볼
# P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
#    예) 현재 배치 데이터의 최대 크기가 4 인 경우
#       word -> ['w', 'o', 'r', 'd']
#       to   -> ['t', 'o', 'P', 'P']


char_pool = ""
for en in range(len(lines_en)):
    char_pool = char_pool + lines_en[en]
for ko in range(len(lines_ko)):
    char_pool = char_pool + lines_ko[ko]
######################################################################
#num_dic = {n: i for i, n in enumerate(char_arr)}

char_dic = {}
char_list = []
char_list.append("S")
char_list.append("E")
char_list.append("P")
char_list.append(" ")
char_list.append("#")
char_list.append("$")

for char1 in char_pool:
    if not (char1 in char_list):
        char_list.append(char1)

char_list.append("`")
char_list.append("%")
char_list.append("^")




for num_value, char2 in enumerate(char_list):
    if not (char2 in char_dic):
        char_dic[char2] = num_value

dic_len = len(char_dic)

print(char_dic);
#print(dic_len);
#print(len(char_pool))


"""
a = False
b = False
if "`" in char_dic :
    a = True
if "" in char_dic:
    b = True
print( a ,  b)
"""




# 영어를 한글로 번역하기 위한 학습 데이터
#seq_data = [['word', '단 '], ['wood', '나 '],
#            ['game', '놀 '], ['girl', '소 '],
#            ['ki  ', '키 '], ['love', '사 ']]


def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 배열로 만든다.
        input = [char_dic[n] for n in seq[0]]
        # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
        output = [char_dic[n] for n in ('$' + seq[1])]
        # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
        target = [char_dic[n] for n in (seq[1]+ '#')]


        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        # 출력값만 one-hot 인코딩이 아님 (sparse_softmax_cross_entropy_with_logits 사용)
        target_batch.append(target)

    return input_batch, output_batch, target_batch


#########
# 옵션 설정
######
learning_rate = 0.0001
n_hidden = 512
total_epoch = 10000
# 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
n_class = n_input = dic_len


#########
# 신경망 모델 구성
######
global_step = tf.Variable(0, trainable=False, name='global_step')

# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
# [batch size, time steps, input size]
enc_input = tf.placeholder(tf.float64, [None, None, n_input])
dec_input = tf.placeholder(tf.float64, [None, None, n_input])
# [batch size, time steps]
targets = tf.placeholder(tf.int32, [None, None])


# 인코더 셀을 구성한다.
with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    #enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.1)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float64)


# 디코더 셀을 구성한다.
with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    #dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.1)

    # Seq2Seq 모델은 인코더 셀의 최종 상태값을
    # 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심.
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float64)


model = tf.layers.dense(outputs, n_class, activation=None)


cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


#########
# 신경망 모델 학습
######
sess = tf.Session()

saver = tf.train.Saver(tf.global_variables()) # 모델 저장을 위한 선언

ckpt = tf.train.get_checkpoint_state('./')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path): # 저장된 모델이 있으면 활용
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))
    saver.save(sess, './model.ckpt', global_step=global_step)
print('최적화 완료!')



#########
# 번역 테스트
######
# 단어를 입력받아 번역 단어를 예측하고 디코딩하는 함수
def translate(word):
    # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
    # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P 값으로 채운다.
    # ['word', 'PPPP']
    seq_data = [word, ' ' * 256]

    input_batch, output_batch, target_batch = make_batch([seq_data])

    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [char_list[i] for i in result[0]]

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    #end = decoded.index('#')
    #translated = ''.join(decoded[:end])
    translated = ''.join(decoded)

    return translated


print('\n=== 번역 테스트 ===')
print('문장1 ->', translate("{0:<512}".format('Mat 1:1 A record of the genealogy of Jesus Christ the son of David, the son of Abraham')))
print('문장2 ->', translate("{0:<512}".format('Gen 1:1 In the beginning God created the heavens and the earth.')))

