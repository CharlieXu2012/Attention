import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.regularizers import l2
from keras.utils import np_utils
from experiments import get_cnf_mat
import keras.backend as K
from os.path import join, expanduser, exists
import os
from keras import regularizers

def DNN_single(shape0):
    input1 = keras.layers.Input(shape=(shape0,))

    x1 = keras.layers.Dense(256,activation='relu')(input1)
    d1 = keras.layers.Dropout(0.2)(x1)

    y1 = keras.layers.Dense(256,activation='relu')(d1)
    d21 = keras.layers.Dropout(0.2)(y1)

    z1 = keras.layers.Dense(256,activation='relu')(d21)
    d31 = keras.layers.Dropout(0.2)(z1)

    fusion = keras.layers.Dense(64,activation='relu')(d31)
    out = keras.layers.Dense(3, activation='softmax')(fusion)

    model = keras.models.Model(inputs=[input1], outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])
    model.summary()
    return model

def early_DNN2(shape0,shape1,fusiontype):
    if fusiontype==1:
        input1 = keras.layers.Input(shape=(shape0,))
        input2 = keras.layers.Input(shape=(shape1,))

        x1 = keras.layers.Dense(256,activation='relu')(input1)
        d1 = keras.layers.Dropout(0.2)(x1)

        x2 = keras.layers.Dense(256,activation='relu')(input2)
        d2 = keras.layers.Dropout(0.2)(x2)


        y1 = keras.layers.Dense(256,activation='relu')(d1)
        d21 = keras.layers.Dropout(0.2)(y1)

        y2 = keras.layers.Dense(256,activation='relu')(d2)
        d22 = keras.layers.Dropout(0.2)(y2)


        z1 = keras.layers.Dense(256,activation='relu')(d21)
        d31 = keras.layers.Dropout(0.2)(z1)

        z2 = keras.layers.Dense(256,activation='relu')(d22)
        d32 = keras.layers.Dropout(0.2)(z2)


        fusion_pre = keras.layers.Concatenate()([d31,d32])

        fusion = keras.layers.Dense(64,activation='relu')(fusion_pre)
        out = keras.layers.Dense(3, activation='softmax')(fusion)

        model = keras.models.Model(inputs=[input1, input2], outputs=out)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    else:
        input1 = keras.layers.Input(shape=(shape0,))
        input2 = keras.layers.Input(shape=(shape1,))
        concat = keras.layers.Concatenate()([input1,input2])

        x1 = keras.layers.Dense(256,activation='relu')(concat)
        d1 = keras.layers.Dropout(0.2)(x1)

        y1 = keras.layers.Dense(256,activation='relu')(d1)
        d21 = keras.layers.Dropout(0.2)(y1)

        z1 = keras.layers.Dense(256,activation='relu')(d21)
        d31 = keras.layers.Dropout(0.2)(z1)

        fusion = keras.layers.Dense(64,activation='relu')(d31)
        out = keras.layers.Dense(3, activation='softmax')(fusion)

        model = keras.models.Model(inputs=[input1, input2], outputs=out)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    model.summary()

    return model

def late_DNN2(shape0,shape1,type):
    if type==0:
        kp = keras.models.load_model(join(os.path.dirname(__file__), 'models\\keypoints.h5'))
        kp.name = 'kp'
        for i in range(0,len(kp.layers)):
            kp.layers[i].name = kp.layers[i].name+ '_kp'
        kp.trainable=False

        dist = keras.models.load_model(join(os.path.dirname(__file__), 'models\\distances.h5'))
        dist.name = 'dist'
        for i in range(0,len(dist.layers)):
            dist.layers[i].name = dist.layers[i].name+ '_dist'
        dist.trainable=False

        newout = keras.layers.Average()([kp.layers[-1].output,dist.layers[-1].output])

        model = keras.models.Model(inputs=[kp.layers[0].input,dist.layers[0].input], outputs=newout)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    elif type==1:
        kp = keras.models.load_model(join(os.path.dirname(__file__), 'models\\keypoints.h5'))
        kp.name = 'kp'
        for i in range(0,len(kp.layers)):
            kp.layers[i].name = kp.layers[i].name+ '_kp'

        dist = keras.models.load_model(join(os.path.dirname(__file__), 'models\\distances.h5'))
        dist.name = 'dist'
        for i in range(0,len(dist.layers)):
            dist.layers[i].name = dist.layers[i].name+ '_dist'

        newout = keras.layers.Maximum()([kp.layers[-1].output,dist.layers[-1].output])

        model = keras.models.Model(inputs=[kp.layers[0].input,dist.layers[0].input], outputs=newout)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    else:
        kp = keras.models.load_model(join(os.path.dirname(__file__), 'models\\keypoints.h5'))
        kp.name = 'kp'
        for i in range(0,len(kp.layers)):
            kp.layers[i].name = kp.layers[i].name+ '_kp'
        kp.trainable=False

        dist = keras.models.load_model(join(os.path.dirname(__file__), 'models\\distances.h5'))
        dist.name = 'dist'
        for i in range(0,len(dist.layers)):
            dist.layers[i].name = dist.layers[i].name+ '_dist'
        dist.trainable=False

        concat = keras.layers.Concatenate()([kp.layers[-1].output,dist.layers[-1].output])
        merge_softmax= keras.layers.Dense(2, activation='softmax')(concat)
        merge_softmax = keras.layers.Reshape((1, 2))(merge_softmax)

        bag_of_models = keras.layers.Concatenate()([kp.layers[-1].output,dist.layers[-1].output])
        bag_of_models = keras.layers.Reshape((2, 3))(bag_of_models)

        final_result = keras.layers.Dot(axes = [1, 2])([bag_of_models,merge_softmax])
        final_result = keras.layers.Reshape((3, ))(final_result)

        model = keras.models.Model(inputs=[kp.layers[0].input,dist.layers[0].input], outputs=final_result)

        for i in range(0,16):
            model.layers[i].trainable = False

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    model.summary()
    return model


def early_DNN3(shape0,shape1,shape2,fusiontype):
    if fusiontype==0:
        input1 = keras.layers.Input(shape=(shape0,))
        input2 = keras.layers.Input(shape=(shape1,))
        input3 = keras.layers.Input(shape=(shape2,))
        concat = keras.layers.Concatenate()([input1,input2,input3])

        x1 = keras.layers.Dense(256,activation='relu')(concat)
        d1 = keras.layers.Dropout(0.2)(x1)


        y1 = keras.layers.Dense(256,activation='relu')(d1)
        d21 = keras.layers.Dropout(0.2)(y1)


        z1 = keras.layers.Dense(256,activation='relu')(d21)
        d31 = keras.layers.Dropout(0.2)(z1)


        fusion = keras.layers.Dense(64,activation='relu')(d31)
        out = keras.layers.Dense(3, activation='softmax')(fusion)

        model = keras.models.Model(inputs=[input1, input2, input3], outputs=out)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    else:
        input1 = keras.layers.Input(shape=(shape0,))
        input2 = keras.layers.Input(shape=(shape1,))
        input3 = keras.layers.Input(shape=(shape2,))

        x1 = keras.layers.Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.001))(input1)
        d1 = keras.layers.Dropout(0.5)(x1)

        x2 = keras.layers.Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.001))(input2)
        d2 = keras.layers.Dropout(0.5)(x2)

        x3 = keras.layers.Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.001))(input3)
        d3 = keras.layers.Dropout(0.5)(x3)


        y1 = keras.layers.Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.001))(d1)
        d21 = keras.layers.Dropout(0.5)(y1)

        y2 = keras.layers.Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.001))(d2)
        d22 = keras.layers.Dropout(0.5)(y2)

        y3 = keras.layers.Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.001))(d3)
        d23 = keras.layers.Dropout(0.5)(y3)


        z1 = keras.layers.Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.001))(d21)
        d31 = keras.layers.Dropout(0.5)(z1)

        z2 = keras.layers.Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.001))(d22)
        d32 = keras.layers.Dropout(0.5)(z2)

        z3 = keras.layers.Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.001))(d23)
        d33 = keras.layers.Dropout(0.5)(z3)

        fusion_pre = keras.layers.Concatenate()([d31,d32,d33])

        fusion = keras.layers.Dense(64,activation='relu', kernel_regularizer=regularizers.l2(0.01))(fusion_pre)
        out = keras.layers.Dense(3, activation='softmax')(fusion)

        model = keras.models.Model(inputs=[input1, input2, input3], outputs=out)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    model.summary()
    return model

def late_DNN3(shape0,shape1,shape2,type):
    if type==0:
        kp = keras.models.load_model(join(os.path.dirname(__file__), 'models\\keypoints.h5'))
        kp.name = 'kp'
        for i in range(0,len(kp.layers)):
            kp.layers[i].name = kp.layers[i].name+ '_kp'
        kp.trainable=False
        dist = keras.models.load_model(join(os.path.dirname(__file__), 'models\\distances.h5'))
        dist.name = 'dist'
        for i in range(0,len(dist.layers)):
            dist.layers[i].name = dist.layers[i].name+ '_dist'
        dist.trainable=False
        dp = keras.models.load_model(join(os.path.dirname(__file__), 'models\\depth.h5'))
        dp.name = 'dp'
        for i in range(0,len(dp.layers)):
            dp.layers[i].name = dp.layers[i].name+ '_dp'
        dp.trainable=False

        newout = keras.layers.Average()([kp.layers[-1].output,dist.layers[-1].output,dp.layers[-1].output])

        model = keras.models.Model(inputs=[kp.layers[0].input,dist.layers[0].input,dp.layers[0].input], outputs=newout)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    elif type==1:
        kp = keras.models.load_model(join(os.path.dirname(__file__), 'models\\keypoints.h5'))
        kp.name = 'kp'
        for i in range(0,len(kp.layers)):
            kp.layers[i].name = kp.layers[i].name+ '_kp'

        dist = keras.models.load_model(join(os.path.dirname(__file__), 'models\\distances.h5'))
        dist.name = 'dist'
        for i in range(0,len(dist.layers)):
            dist.layers[i].name = dist.layers[i].name+ '_dist'

        dp = keras.models.load_model(join(os.path.dirname(__file__), 'models\\depth.h5'))
        dp.name = 'dp'
        for i in range(0,len(dp.layers)):
            dp.layers[i].name = dp.layers[i].name+ '_dp'

        newout = keras.layers.Maximum()([kp.layers[-1].output,dist.layers[-1].output,dp.layers[-1].output])

        model = keras.models.Model(inputs=[kp.layers[0].input,dist.layers[0].input,dp.layers[0].input], outputs=newout)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    else:
        kp = keras.models.load_model(join(os.path.dirname(__file__), 'models\\keypoints.h5'))
        kp.name = 'kp'
        for i in range(0,len(kp.layers)):
            kp.layers[i].name = kp.layers[i].name+ '_kp'

        dist = keras.models.load_model(join(os.path.dirname(__file__), 'models\\distances.h5'))
        dist.name = 'dist'
        for i in range(0,len(dist.layers)):
            dist.layers[i].name = dist.layers[i].name+ '_dist'

        dp = keras.models.load_model(join(os.path.dirname(__file__), 'models\\depth.h5'))
        dp.name = 'dp'
        for i in range(0,len(dp.layers)):
            dp.layers[i].name = dp.layers[i].name+ '_dp'

        concat = keras.layers.Concatenate()([kp.layers[-1].output,dist.layers[-1].output,dp.layers[-1].output])
        merge_softmax= keras.layers.Dense(2, activation='softmax')(concat)
        merge_softmax = keras.layers.Reshape((1, 2))(merge_softmax)

        bag_of_models = keras.layers.Concatenate()([kp.layers[-1].output,dist.layers[-1].output])
        bag_of_models = keras.layers.Reshape((2, 3))(bag_of_models)

        final_result = keras.layers.Dot(axes = [1, 2])([bag_of_models,merge_softmax])
        final_result = keras.layers.Reshape((3, ))(final_result)

        model = keras.models.Model(inputs=[kp.layers[0].input,dist.layers[0].input,dp.layers[0].input], outputs=final_result)
        for i in range(0,24):
            model.layers[i].trainable = False

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    model.summary()
    return model


def evaluate_flexible(model, X_train, Y_train, X_test, Y_test, X_depth_train, X_depth_test, modelshape,bs,ep):
    #tboard = keras.callbacks.TensorBoard(log_dir='./logs2', histogram_freq=5, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    if modelshape==2:

        history = model.fit([np.concatenate([X_train[:,0:12], X_train[:,26:54]],1), np.concatenate([X_train[:,12:24], X_train[:,54:66]],1)], np_utils.to_categorical(Y_train,num_classes=3), 
                 batch_size=bs, nb_epoch=ep,validation_data=([np.concatenate([X_test[:,0:12], X_test[:,26:54]],1), np.concatenate([X_test[:,12:24], X_test[:,54:66]],1)], np_utils.to_categorical(Y_test,num_classes=3)),verbose=2)

        pred = model.predict([np.concatenate([X_test[:,0:12], X_test[:,26:54]],1), np.concatenate([X_test[:,12:24], X_test[:,54:66]],1)], batch_size=32, verbose=2, steps=None)
        class_pred = pred.argmax(axis=-1)
        cnf_matrix = get_cnf_mat(Y_test,class_pred)

    elif modelshape==3:
        history = model.fit([np.concatenate([X_train[:,0:12], X_train[:,26:54]],1), np.concatenate([X_train[:,12:24], X_train[:,54:66]],1),X_depth_train], np_utils.to_categorical(Y_train,num_classes=3), 
                 batch_size=bs, nb_epoch=ep,validation_data=([np.concatenate([X_test[:,0:12], X_test[:,26:54]],1), np.concatenate([X_test[:,12:24], X_test[:,54:66]],1),X_depth_test], np_utils.to_categorical(Y_test,num_classes=3)),verbose=2)

        pred = model.predict([np.concatenate([X_test[:,0:12], X_test[:,26:54]],1), np.concatenate([X_test[:,12:24], X_test[:,54:66]],1),X_depth_test], batch_size=32, verbose=2, steps=None)
        class_pred = pred.argmax(axis=-1)
        cnf_matrix = get_cnf_mat(Y_test,class_pred)

    if modelshape==1:
        history = model.fit([X_train], np_utils.to_categorical(Y_train,num_classes=3), 
                 batch_size=bs, nb_epoch=ep,validation_data=([X_test], np_utils.to_categorical(Y_test,num_classes=3)),verbose=2)

        pred = model.predict([X_test], batch_size=32, verbose=2, steps=None)
        class_pred = pred.argmax(axis=-1)
        cnf_matrix = get_cnf_mat(Y_test,class_pred)

    return history, pred, cnf_matrix, model
