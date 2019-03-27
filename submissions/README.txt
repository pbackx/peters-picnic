21-03-19.tsv: 
Resnet50
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(labels), activation='softmax'))

Trained for about 100 epochs (maybe less?)

26-03-19.tsv:
Same as above, but trained for 200 epochs
Things were still improving, so maybe train longer?

