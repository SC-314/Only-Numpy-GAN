np.random.seed(42)
random.seed(42)

for _ in range(10000):
    total += 1 
    ########### get the images ##############
    batch_size = 1
    real_images = []
    for i in range(batch_size): # add the random real images
        real_images.append(random.choice(data).reshape((1, 28, 28)))
    noise_list = list(np.random.normal(0,1, (batch_size, 100))) # add the noise vectors for the generators

    fake_images = []
    fake_gen_data = []
    for i in noise_list: # calculate the generated images
        fake_temp = gen(i, GW0,  GW1, GW2, GW3, GW4, GW5, GW6)
        fake_images.append(fake_temp[0]) # add the fake images, and the data
        fake_gen_data.append(list(fake_temp))
    
    ########### output the discriminators ##############
    real_outputs = []
    fake_outputs = []
    real_dis_data = []
    fake_dis_data = []
    for i in real_images:
        real_temp = dis(i, DW0, DW2, DW4) # use the discriminator on the real images
        real_outputs.append(real_temp[0])
        real_dis_data.append(list(real_temp))
    
    for i in fake_images:
        fake_temp = dis(i, DW0, DW2, DW4) # use the generator on the real images
        fake_outputs.append(fake_temp[0])
        fake_dis_data.append(list(fake_temp))
    
    ########### do the loss function ##############
    epsilon = 1e-10 # loss of the discriminator
    d_loss_real = -np.mean(np.log(np.array(real_outputs)+epsilon))
    d_loss_fake = -np.mean(np.log(1 - np.array(fake_outputs)+epsilon))
    d_loss = d_loss_real + d_loss_fake
    
    ########### loss derivative ##############
    dLdreal = -1/(batch_size * np.array(real_outputs).reshape(-1)+epsilon) # derivative functions for the discriminator on
    dLdfake = 1/(batch_size * (1 - np.array(fake_outputs).reshape(-1))+epsilon) # the real and fake images
    dLdtotal = dLdreal + dLdfake
    
    ########### discriminator back prop ##############
    dLdDW4, dLdDW2, dLdDW0 = 0, 0, 0
    for i in range(batch_size):
        fake_dis_data[i].insert(0, dLdfake[i]) # DLD, DZ4, DX4, DW4, DZ3, DX3, DX3M, DX2, DW2, DX1, DX1M, DX0, DW0
        real_dis_data[i].insert(0, dLdreal[i])
    for i in range(batch_size):
        temp0, temp1, temp2 = dis_bp(fake_dis_data[i]) # Back-propagate through the discriminator, (images from generator)
        dLdDW4 += temp0/batch_size
        dLdDW2 += temp1/batch_size
        dLdDW0 += temp2/batch_size
    
    for i in range(batch_size):
        temp0, temp1, temp2 = dis_bp(real_dis_data[i]) # back-propagate through the discriminator, (real images)
        dLdDW4 += temp0/batch_size
        dLdDW2 += temp1/batch_size
        dLdDW0 += temp2/batch_size
    
    ########### generator back prop ##############
    dLdGW0, dLdGW1, dLdGW2, dLdGW3, dLdGW4, dLdGW5, dLdGW6 = 0, 0, 0, 0, 0, 0, 0
    g_loss = -np.mean(np.log(np.array(fake_outputs).reshape(-1))) # loss and loss derivative of the generator
    dLdfake_gen_DG = -1/(batch_size * np.array(fake_outputs).reshape(-1)+epsilon)
    
    for i in range(batch_size):
        fake_dis_data[i][0] = dLdfake_gen_DG[i]
    
    dLdfake_gen_G = []
    for i in range(batch_size): # back-propagate through the generator
        dLdfake_gen_G.append(dis_gen_bp(fake_dis_data[i]))
    
    
    for i in range(batch_size):
        fake_gen_data[i].insert(0, dLdfake_gen_G[i])
    
    for i in range(batch_size): # sum the weight gradients over the batch
        temp0, temp1, temp2, temp3, temp4, temp5, temp6 = gen_bp(fake_gen_data[i])

        dLdGW6 += temp0/batch_size
        dLdGW5 += temp1/batch_size
        dLdGW4 += temp2/batch_size
        dLdGW3 += temp3/batch_size
        dLdGW2 += temp4/batch_size
        dLdGW1 += temp5/batch_size
        dLdGW0 += temp6/batch_size
    
    # Altar the weights with different learning rates
    LR = 0.01 * ((5/6) ** (total/1000)) * 0.5
    LR1 = 0.01 * ((5/6) ** (total/1000)) * 0.5
    DW0 -= LR1 * np.clip(dLdDW0, -5, 5)
    DW2 -= LR1 * np.clip(dLdDW2, -5, 5)
    DW4 -= LR1 * np.clip(dLdDW4, -5, 5)
    
    GW0 -= LR * np.clip(dLdGW0, -5, 5)
    GW1 -= LR * np.clip(dLdGW1, -5, 5)
    GW2 -= LR * np.clip(dLdGW2, -5, 5)
    GW3 -= LR * np.clip(dLdGW3, -5, 5)
    GW4 -= LR * np.clip(dLdGW4, -5, 5)
    GW5 -= LR * np.clip(dLdGW5, -5, 5)
    GW6 -= LR * np.clip(dLdGW6, -5, 5)

    ########### storing and graphing data ##############
    if _ % 100 == 0:
        np.savez(f'final_weights{_}.npz', 
         DW0=DW0, 
         DW2=DW2, 
         DW4=DW4, 
         GW0=GW0, 
         GW1=GW1, 
         GW2=GW2, 
         GW3=GW3,
         GW4=GW4,
         GW5=GW5,
         GW6=GW6)

    
    if _ % 10 == 0:
        print("HELLO")
        temp_data = []
        for i in test_noise:
            temp_data.append(gen(i, GW0,  GW1, GW2, GW3, GW4, GW5, GW6))
            
        store_constant_images.append(temp_data)

        clear_output(wait=True)
        plt.imshow(Image.fromarray(np.array((fake_images[0]+0.15) * 255, dtype=np.uint8).reshape((28,28))))
        plt.show()
        G.append(g_loss)
        D.append(d_loss)
        plt.plot(G)
        plt.plot(D)
        #plt.ylim(0,5)
        plt.text(0, 0.5, 'g_loss', color='blue', fontsize=12)
        plt.text(0, 1.5, 'd_loss', color='orange', fontsize=12)
        plt.show()

    store_fake_images.append(fake_images)
    print(g_loss, d_loss)
