import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.regularizers import l2
from keras.utils import np_utils
from experiments import get_cnf_mat
import keras.backend as K

"""
def simple_LSTM(shape):
    input1 = keras.layers.Input(shape=(shape,38,))
    input2 = keras.layers.Input(shape=(shape,6,))
    
    concat = keras.layers.Concatenate()([input1,input2])

    x1 = keras.layers.LSTM(256,return_sequences=False,
                            activation='relu',
                            dropout=0.5)(concat)

    #y1 = keras.layers.LSTM(256,return_sequences=False,activation='tanh',recurrent_dropout=0.3,dropout=0.5)(x1)

    d1 = keras.layers.Dense(64,activation='relu')(x1)
    out = keras.layers.Dense(3, activation='softmax')(d1)

    model = keras.models.Model(inputs=[input1, input2], outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    return model
"""
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
        out1 = keras.layers.Dense(3, activation='softmax')(d31)

        z2 = keras.layers.Dense(256,activation='relu')(d22)
        d32 = keras.layers.Dropout(0.2)(z2)
        out2 = keras.layers.Dense(3, activation='softmax')(d32)

        fusion = keras.layers.Average()([out1,out2])
    
        model = keras.models.Model(inputs=[input1, input2], outputs=fusion)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    elif type==1:
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
        out1 = keras.layers.Dense(3, activation='softmax')(d31)

        z2 = keras.layers.Dense(256,activation='relu')(d22)
        d32 = keras.layers.Dropout(0.2)(z2)
        out2 = keras.layers.Dense(3, activation='softmax')(d32)

        fusion = keras.layers.Maximum()([out1,out2])
    
        model = keras.models.Model(inputs=[input1, input2], outputs=fusion)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    else:
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
        out1 = keras.layers.Dense(3, activation='softmax')(d31)

        z2 = keras.layers.Dense(256,activation='relu')(d22)
        d32 = keras.layers.Dropout(0.2)(z2)
        out2 = keras.layers.Dense(3, activation='softmax')(d32)


        concat = keras.layers.Concatenate()([out1,out2])
        merge_softmax= keras.layers.Dense(2, activation='softmax')(concat)
        merge_softmax = keras.layers.Reshape((1, 2))(merge_softmax)

        bag_of_models = keras.layers.Concatenate()([out1,out2])
        bag_of_models = keras.layers.Reshape((2, 3))(bag_of_models)

        final_result = keras.layers.Dot(axes = [1, 2])([bag_of_models,merge_softmax])
        final_result = keras.layers.Reshape((3, ))(final_result)

        model = keras.models.Model(inputs=[input1, input2], outputs=final_result)

        #fusion = keras.layers.Average()([out1,out2])
    
        #model = keras.models.Model(inputs=[input1, input2], outputs=fusion)

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

        x1 = keras.layers.Dense(256,activation='relu')(input1)
        d1 = keras.layers.Dropout(0.2)(x1)

        x2 = keras.layers.Dense(256,activation='relu')(input2)
        d2 = keras.layers.Dropout(0.2)(x2)

        x3 = keras.layers.Dense(256,activation='relu')(input3)
        d3 = keras.layers.Dropout(0.2)(x3)


        y1 = keras.layers.Dense(256,activation='relu')(d1)
        d21 = keras.layers.Dropout(0.2)(y1)

        y2 = keras.layers.Dense(256,activation='relu')(d2)
        d22 = keras.layers.Dropout(0.2)(y2)

        y3 = keras.layers.Dense(256,activation='relu')(d3)
        d23 = keras.layers.Dropout(0.2)(y3)


        z1 = keras.layers.Dense(256,activation='relu')(d21)
        d31 = keras.layers.Dropout(0.2)(z1)

        z2 = keras.layers.Dense(256,activation='relu')(d22)
        d32 = keras.layers.Dropout(0.2)(z2)

        z3 = keras.layers.Dense(256,activation='relu')(d23)
        d33 = keras.layers.Dropout(0.2)(z3)

        fusion_pre = keras.layers.Concatenate()([d31,d32,d33])

        fusion = keras.layers.Dense(64,activation='relu')(fusion_pre)
        out = keras.layers.Dense(3, activation='softmax')(fusion)

        model = keras.models.Model(inputs=[input1, input2, input3], outputs=out)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    model.summary()
    return model

def late_DNN3(shape0,shape1,shape2,type):
    if type==0:
        input1 = keras.layers.Input(shape=(shape0,))
        input2 = keras.layers.Input(shape=(shape1,))
        input3 = keras.layers.Input(shape=(shape2,))

        x1 = keras.layers.Dense(256,activation='relu')(input1)
        d1 = keras.layers.Dropout(0.2)(x1)

        x2 = keras.layers.Dense(256,activation='relu')(input2)
        d2 = keras.layers.Dropout(0.2)(x2)

        x3 = keras.layers.Dense(256,activation='relu')(input3)
        d3 = keras.layers.Dropout(0.2)(x3)


        y1 = keras.layers.Dense(256,activation='relu')(d1)
        d21 = keras.layers.Dropout(0.2)(y1)

        y2 = keras.layers.Dense(256,activation='relu')(d2)
        d22 = keras.layers.Dropout(0.2)(y2)

        y3 = keras.layers.Dense(256,activation='relu')(d3)
        d23 = keras.layers.Dropout(0.2)(y3)


        z1 = keras.layers.Dense(256,activation='relu')(d21)
        d31 = keras.layers.Dropout(0.2)(z1)
        out1 = keras.layers.Dense(3, activation='softmax')(d31)

        z2 = keras.layers.Dense(256,activation='relu')(d22)
        d32 = keras.layers.Dropout(0.2)(z2)
        out2 = keras.layers.Dense(3, activation='softmax')(d32)

        z3 = keras.layers.Dense(256,activation='relu')(d23)
        d33 = keras.layers.Dropout(0.2)(z3)
        out3 = keras.layers.Dense(3, activation='softmax')(d33)

        fusion = keras.layers.Average([out1,out2,out3])

        model = keras.models.Model(inputs=[input1, input2, input3], outputs=fusion)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    elif type==1:
        input1 = keras.layers.Input(shape=(shape0,))
        input2 = keras.layers.Input(shape=(shape1,))
        input3 = keras.layers.Input(shape=(shape2,))

        x1 = keras.layers.Dense(256,activation='relu')(input1)
        d1 = keras.layers.Dropout(0.2)(x1)

        x2 = keras.layers.Dense(256,activation='relu')(input2)
        d2 = keras.layers.Dropout(0.2)(x2)

        x3 = keras.layers.Dense(256,activation='relu')(input3)
        d3 = keras.layers.Dropout(0.2)(x3)


        y1 = keras.layers.Dense(256,activation='relu')(d1)
        d21 = keras.layers.Dropout(0.2)(y1)

        y2 = keras.layers.Dense(256,activation='relu')(d2)
        d22 = keras.layers.Dropout(0.2)(y2)

        y3 = keras.layers.Dense(256,activation='relu')(d3)
        d23 = keras.layers.Dropout(0.2)(y3)


        z1 = keras.layers.Dense(256,activation='relu')(d21)
        d31 = keras.layers.Dropout(0.2)(z1)
        out1 = keras.layers.Dense(3, activation='softmax')(d31)

        z2 = keras.layers.Dense(256,activation='relu')(d22)
        d32 = keras.layers.Dropout(0.2)(z2)
        out2 = keras.layers.Dense(3, activation='softmax')(d32)

        z3 = keras.layers.Dense(256,activation='relu')(d23)
        d33 = keras.layers.Dropout(0.2)(z3)
        out3 = keras.layers.Dense(3, activation='softmax')(d33)

        fusion = keras.layers.Maximum([out1,out2,out3])

        model = keras.models.Model(inputs=[input1, input2, input3], outputs=fusion)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    else:
        input1 = keras.layers.Input(shape=(shape0,))
        input2 = keras.layers.Input(shape=(shape1,))
        input3 = keras.layers.Input(shape=(shape2,))

        x1 = keras.layers.Dense(256,activation='relu')(input1)
        d1 = keras.layers.Dropout(0.2)(x1)

        x2 = keras.layers.Dense(256,activation='relu')(input2)
        d2 = keras.layers.Dropout(0.2)(x2)

        x3 = keras.layers.Dense(256,activation='relu')(input3)
        d3 = keras.layers.Dropout(0.2)(x3)


        y1 = keras.layers.Dense(256,activation='relu')(d1)
        d21 = keras.layers.Dropout(0.2)(y1)

        y2 = keras.layers.Dense(256,activation='relu')(d2)
        d22 = keras.layers.Dropout(0.2)(y2)

        y3 = keras.layers.Dense(256,activation='relu')(d3)
        d23 = keras.layers.Dropout(0.2)(y3)


        z1 = keras.layers.Dense(256,activation='relu')(d21)
        d31 = keras.layers.Dropout(0.2)(z1)
        out1 = keras.layers.Dense(3, activation='softmax')(d31)

        z2 = keras.layers.Dense(256,activation='relu')(d22)
        d32 = keras.layers.Dropout(0.2)(z2)
        out2 = keras.layers.Dense(3, activation='softmax')(d32)

        z3 = keras.layers.Dense(256,activation='relu')(d23)
        d33 = keras.layers.Dropout(0.2)(z3)
        out3 = keras.layers.Dense(3, activation='softmax')(d33)

        concat = keras.layers.Concatenate()([out1,out2,out3])
        merge_softmax= keras.layers.Dense(3, activation='softmax')(concat)
        merge_softmax = keras.layers.Reshape((1, 3))(merge_softmax)

        bag_of_models = keras.layers.Concatenate()([out1,out2,out3])
        bag_of_models = keras.layers.Reshape((3, 3))(bag_of_models)

        final_result = keras.layers.Dot(axes = [1, 2])([bag_of_models,merge_softmax])
        final_result = keras.layers.Reshape((3, ))(final_result)

        model = keras.models.Model(inputs=[input1, input2, input3], outputs=final_result)

        #fusion = keras.layers.Average([out1,out2,out3])

        #model = keras.models.Model(inputs=[input1, input2, input3], outputs=fusion)

        model.compile(loss='categorical_crossentropy',
                      optimizer = keras.optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])
    model.summary()
    return model

def evaluate_lstm(model, train, gt_train, test, 
                  gt_test, depth_train, depth_test, depth_label, simple):
    
    if simple == True:
        history = model.fit([train, depth_train], np_utils.to_categorical(gt_train,num_classes=3), 
                 batch_size=16, nb_epoch=125,validation_data=([test, depth_test], np_utils.to_categorical(gt_test,num_classes=3)),verbose=2,shuffle=False)
        pred = model.predict([test, depth_test], batch_size=16, verbose=2, steps=None)
        class_pred = pred.argmax(axis=-1)
        cnf_matrix = get_cnf_mat(gt_test,class_pred)
    else:
        if depth_label==True:
            history = model.fit([train[:,:,0:12], train[:,:,12:24], train[:,:,24:26], train[:,:,26:40], train[:,:,40:54], train[:,:,54:66], depth_train], np_utils.to_categorical(gt_train,num_classes=3), 
                     batch_size=32, nb_epoch=75,validation_data=([test[:,:,0:12], test[:,:,12:24], test[:,:,24:26], test[:,:,26:40], test[:,:,40:54], test[:,:,54:66], depth_test], np_utils.to_categorical(gt_test,num_classes=3)),verbose=2)
        
            pred = model.predict([test[:,:,0:12], test[:,:,12:24], test[:,:,24:26], test[:,:,26:40], test[:,:,40:54], test[:,:,54:66], test[:,:,66:72], depth_test], batch_size=32, verbose=2, steps=None)
            class_pred = pred.argmax(axis=-1)
            cnf_matrix = get_cnf_mat(gt_test,class_pred)
        else:
            history = model.fit([train[:,:,0:12], train[:,:,12:24], train[:,:,24:26], train[:,:,26:40], train[:,:,40:54], train[:,:,54:66]], np_utils.to_categorical(gt_train,num_classes=3), 
                     batch_size=16, nb_epoch=75,validation_data=([test[:,:,0:12], test[:,:,12:24], test[:,:,24:26], test[:,:,26:40], test[:,:,40:54], test[:,:,54:66]], np_utils.to_categorical(gt_test,num_classes=3)),verbose=2)

            pred = model.predict([test[:,:,0:12], test[:,:,12:24], test[:,:,24:26], test[:,:,26:40], test[:,:,40:54], test[:,:,54:66]], batch_size=32, verbose=2, steps=None)
            class_pred = pred.argmax(axis=-1)
            cnf_matrix = get_cnf_mat(gt_test,class_pred)

    return history, pred, cnf_matrix

def evaluate_flexible(model, X_train, Y_train, X_test, Y_test, X_depth_train, X_depth_test, modelshape,bs,ep):
    
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
        history = model.fit([X_depth_train], np_utils.to_categorical(Y_train,num_classes=3), 
                 batch_size=bs, nb_epoch=ep,validation_data=([X_depth_test], np_utils.to_categorical(Y_test,num_classes=3)),verbose=2)

        pred = model.predict([X_depth_test], batch_size=32, verbose=2, steps=None)
        class_pred = pred.argmax(axis=-1)
        cnf_matrix = get_cnf_mat(Y_test,class_pred)

    return history, pred, cnf_matrix, model
