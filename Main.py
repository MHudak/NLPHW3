__author__ = 'marc'

import nltk
from nltk.corpus import movie_reviews

tt_division = 500
tt_top = 1200
words = movie_reviews.words()[tt_division:tt_top]
words1 = movie_reviews.words()[:tt_division]
pos = movie_reviews.words(categories='pos')[tt_division:tt_top]
pos1 = movie_reviews.words(categories='pos')[:tt_division]
neg = movie_reviews.words(categories='neg')[tt_division:tt_top]
neg1 = movie_reviews.words(categories='neg')[:tt_division]
neutWords = {}
posWords = {}
negWords = {}
neutWords1 = {}
posWords1 = {}
negWords1 = {}

#words 1
for i in range(0, len(words)):
    neutWords[words[i]] = 1
    posWords[words[i]] = 0
    negWords[words[i]] = 0
print('neutral words added')

for i in range(0, len(pos)):
    posWords[pos[i]] = 1

print('pos words updated')

for i in range(0, len(neg)):
    negWords[neg[i]] = 1

print('neg words updated')
print('done loading dict')

#words2
for i in range(0, len(words1)):
    neutWords1[words1[i]] = 1
    posWords1[words1[i]] = 0
    negWords1[words1[i]] = 0
print('neutral words added')

for i in range(0, len(pos1)):
    posWords1[pos1[i]] = 1

print('pos words updated')

for i in range(0, len(neg1)):
    negWords1[neg1[i]] = 1

print('neg words updated')
print('done loading dict')

tagged = []

train = []

# for k in neutWords:
#     # posword
#     if posWords[k] > 0 & negWords[k] < 1:
#         train.append((dict(a=neutWords[k], b=posWords[k], c=negWords[k]), 'pos'))
#     # negword
#     else:
#         if negWords[k] > 0 & posWords[k] < 1:
#             train.append((dict(a=neutWords[k], b=posWords[k], c=negWords[k]), 'neg'))
#         # negword
#         else:
#             train.append((dict(a=neutWords[k], b=posWords[k], c=negWords[k]), 'neut'))

train = [
    (neutWords, 'neut'),
    (posWords, 'pos'),
    (negWords, 'neg')
]
print(train)
test = []

for k in neutWords1:
    # posword
    if posWords1[k] > 0 & negWords1[k] < 1:
        test.append((dict(a=neutWords1[k], b=posWords1[k], c=negWords1[k])))
    # negword
    else:
        if negWords1[k] > 0 & posWords1[k] < 1:
            train.append((dict(a=neutWords1[k], b=posWords1[k], c=negWords1[k])))
        # negword
        else:
            train.append((dict(a=neutWords1[k], b=posWords1[k], c=negWords1[k])))

# test = [
#     neutWords1,
#     posWords1,
#     negWords1
# ]

# for k in taggedWords:
#     if taggedWords[k] == 'pos':
#         tagged.append(({taggedWords[k]: 1}, 'pos'))
#     else:
#         if taggedWords[k] == 'neg':
#             tagged.append(({taggedWords[k]: 1}, 'neg'))
#         else:
#             tagged.append(({taggedWords[k]: 1}, 'neut'))
#
# print('done converting to array')


# for i in range(0, 25):
#     print(i)
#     if words[i] in pos:
#         taggedWords.append((words[i], 'pos'))
#     else:
#         if words[i] in neg:
#             taggedWords.append((words[i], 'neg'))
#         else:
#             taggedWords.append((words[i], 'neut'))
# print(taggedWords)

classifier = nltk.classify.MaxentClassifier.train(train, 'GIS', trace=0, max_iter=1000)
for featureset in test:
    pdist = classifier.prob_classify(featureset)
    print('%8.2f%6.2f%6.2f' % (pdist.prob('pos'), pdist.prob('neg'), pdist.prob('neut')), end=' ')
print()