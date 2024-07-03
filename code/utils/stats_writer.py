def getSingleConvStats(conv):
    return conv.weight

def getConvStats(model):
    return model[0].weight, model[3].num_dropped_neurons, model[4].weight

def writeDropoutStats(drop_m1, drop_m2,
                        writer, iter, i, name):
    writer.add_scalar(f'NetStats/{i}_{name}/dropout_diff',
                        abs(drop_m1 - drop_m2), iter)
    
def writeConvStats(conv_diff, writer, iter, i, name, conv_id):
        writer.add_scalar(f'NetStats/{i}_{name}/conv{conv_id}_diff',
                        abs(conv_diff), iter)

def writeConvLayerStats(conv1_m1, drop_m1, conv2_m1,
                        conv1_m2, drop_m2, conv2_m2,
                        writer, iter, i, name):
    conv1_diff = (abs(conv1_m1 - conv1_m2)).sum()
    conv2_diff = (abs(conv2_m1 - conv2_m2)).sum()
    conv_diff = conv1_diff + conv2_diff

    writeConvStats(conv1_diff, writer, iter, i, name, 1)
    writeConvStats(conv2_diff, writer, iter, i, name, 2)

    writer.add_scalar(f'NetStats/{i}_{name}/convLayer_diff',
                        abs(conv_diff), iter)
                
    writeDropoutStats(drop_m1, drop_m2, writer, iter, i, name)

def writeNetEncoderStats(model1, model2, writer, iter):
    layers1 = [model1.in_conv.conv_conv,
                model1.down1.maxpool_conv[1].conv_conv,
                model1.down2.maxpool_conv[1].conv_conv,
                model1.down3.maxpool_conv[1].conv_conv,
                model1.down4.maxpool_conv[1].conv_conv,
              ]
    
    layers2 = [model2.in_conv.conv_conv,
                model2.down1.maxpool_conv[1].conv_conv,
                model2.down2.maxpool_conv[1].conv_conv,
                model2.down3.maxpool_conv[1].conv_conv,
                model2.down4.maxpool_conv[1].conv_conv,
              ]

    name = 'encoder'
    for i in range(5):
        conv1_m1, drop_m1, conv2_m1 = getConvStats(layers1[i])
        conv1_m2, drop_m2, conv2_m2 = getConvStats(layers2[i])

        writeConvLayerStats(conv1_m1, drop_m1, conv2_m1,
                            conv1_m2, drop_m2, conv2_m2,
                            writer, iter, i, name)

def writeNetDecoderStats(model1, model2, writer, iter):
    layers1 = [model1.up1.conv1x1,
              model1.up1.conv.conv_conv,
              model1.up2.conv1x1,
              model1.up2.conv.conv_conv,
              model1.up3.conv1x1,
              model1.up3.conv.conv_conv,
              model1.up4.conv1x1,
              model1.up4.conv.conv_conv,
              ]
    
    layers2 = [model2.up1.conv1x1,
              model2.up1.conv.conv_conv,
              model2.up2.conv1x1,
              model2.up2.conv.conv_conv,
              model2.up3.conv1x1,
              model2.up3.conv.conv_conv,
              model2.up4.conv1x1,
              model2.up4.conv.conv_conv,
              ]

    name = 'decoder'
    for i in range(8):
        if i % 2 :
            conv1_m1, drop_m1, conv2_m1 = getConvStats(layers1[i])
            conv1_m2, drop_m2, conv2_m2 = getConvStats(layers2[i])

            writeConvLayerStats(conv1_m1, drop_m1, conv2_m1,
                                conv1_m2, drop_m2, conv2_m2, 
                                writer, iter, i + 5, name)
        else:
            conv_m1 = getSingleConvStats(layers1[i])
            conv_m2 = getSingleConvStats(layers2[i])

            conv1_diff = (abs(conv_m1 - conv_m2)).sum()

            writeConvStats(conv1_diff, writer, iter, i + 5, name, 1)

def writeNetStats(model1, model2, writer, iter):
    # assert type(model1) == type(model2) == UNet

    writeNetEncoderStats(model1.encoder, model2.encoder, writer, iter)
    writeNetDecoderStats(model1.decoder, model2.decoder, writer, iter)