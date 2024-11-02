def dis(GZ2, DW0, DW2, DW4):
    DX0 = pad(GZ2, 1)
    DZ0 = conv(DX0, DW0)
    DX1 = DZ0
    DZ1, DX1M = maxpool(DX1, 2)
    DX2 = pad(DZ1, 1)
    DZ2 = conv(DX2, DW2)
    DX3 = DZ2
    DZ3, DX3M = maxpool(DX3, 2)
    DX4 = DZ3.reshape(1,-1)
    DZ4 = sigmoid(DX4 @ DW4)
    return DZ4, DX4, DW4, DZ3, DX3, DX3M, DX2, DW2, DX1, DX1M, DX0, DW0

def dis_bp(a):
    dL, DZ4, DX4, DW4, DZ3, DX3, DX3M, DX2, DW2, DX1, DX1M, DX0, DW0 = a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12]
    dLdDZ4 = dL * DZ4 * (1 - DZ4)
    dLdDW4  = DX4.reshape(-1,1) @ dLdDZ4
    dLdDX4 = dLdDZ4 @ DW4.T
    
    dLdDZ3 = dLdDX4.reshape(DZ3.shape)
    dLdDX3 = maxpoolBP(DX3, dLdDZ3, DX3M)
    
    dLdDZ2 = dLdDX3
    dLdDW2 = dLdW(DX2, dLdDZ2, DW2)
    dLdDX2 = dLdX(dLdDZ2, DW2, DX2)
    
    dLdDZ1 = pad_bp(dLdDX2, 1)
    dLdDX1 = maxpoolBP(DX1, dLdDZ1, DX1M)

    dLdDZ0 = dLdDX1
    dLdDW0 = dLdW(DX0, dLdDZ0, DW0)

    return dLdDW4, dLdDW2, dLdDW0

def dis_gen_bp(a):
    dL, DZ4, DX4, DW4, DZ3, DX3, DX3M, DX2, DW2, DX1, DX1M, DX0, DW0 = a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12]
    dLdDZ4 = dL * DZ4 * (1 - DZ4)
    dLdDW4  = DX4.reshape(-1,1) @ dLdDZ4
    dLdDX4 = dLdDZ4 @ DW4.T
    
    dLdDZ3 = dLdDX4.reshape(DZ3.shape)
    dLdDX3 = maxpoolBP(DX3, dLdDZ3, DX3M)
    
    dLdDZ2 = dLdDX3
    dLdDW2 = dLdW(DX2, dLdDZ2, DW2)
    dLdDX2 = dLdX(dLdDZ2, DW2, DX2)
    
    dLdDZ1 = pad_bp(dLdDX2, 1)
    dLdDX1 = maxpoolBP(DX1, dLdDZ1, DX1M)

    dLdDZ0 = dLdDX1
    dLdDW0 = dLdW(DX0, dLdDZ0, DW0)
    dLdDX0 = pad_bp(dLdX(dLdDZ0, DW0, DX0), 1)

    return dLdDX0

def gen_bp(a):
    dLdDX0, GZ6,GX6,GW6, GX5,GW5, GX4,GW4, GX3,GW3, GX2,GW2, GX1,GW1, GX0 = a[0], a[1],a[2],a[3], a[4],a[5], a[6],a[7], a[8],a[9], a[10],a[11], a[12],a[13], a[14]

    dLdGZ6 = dLdDX0 * (1 - GZ6 ** 2)
    dLdGW6 = dLdWT(dLdGZ6, GX6, GW6, 1)
    dLdGX6 = dLdXT(dLdGZ6, GW6, GX6, 1)

    dLdGZ5 = dLdGX6 * ReLU_BP(GX6)
    dLdGW5 = dLdWT(dLdGZ5, GX5, GW5, 1)
    dLdGX5 = dLdXT(dLdGZ5, GW5, GX5, 1)

    dLdGZ4 = dLdGX5 * ReLU_BP(GX5)
    dLdGW4 = dLdWT(dLdGZ4, GX4, GW4, 1)
    dLdGX4 = dLdXT(dLdGZ4, GW4, GX4, 1)

    dLdGZ3 = dLdGX4 * ReLU_BP(GX4)
    dLdGW3 = dLdWT(dLdGZ3, GX3, GW3, 1)
    dLdGX3 = dLdXT(dLdGZ3, GW3, GX3, 1)

    dLdGZ2 = dLdGX3 * ReLU_BP(GX3)
    dLdGW2 = dLdWT(dLdGZ2, GX2, GW2, 1)
    dLdGX2 = dLdXT(dLdGZ2, GW2, GX2, 1)

    dLdGZ1 = dLdGX2 * ReLU_BP(GX2)
    dLdGW1 = dLdWT(dLdGZ1, GX1, GW1, 1)
    dLdGX1 = dLdXT(dLdGZ1, GW1, GX1, 1)

    dLdGZ0 = dLdGX1.reshape((1,-1)) * ReLU_BP(GX1.reshape((1,-1)))
    dLdGW0 = GX0.reshape(-1,1) @ dLdGZ0

    return dLdGW6, dLdGW5, dLdGW4, dLdGW3, dLdGW2, dLdGW1, dLdGW0

def gen(GX0, GW0,  GW1, GW2, GW3, GW4, GW5, GW6):
    GZ0 = ReLU(GX0 @ GW0)
    GX1 = GZ0.reshape((-1, 7, 7))
    GZ1 = ReLU(convT(GX1, GW1, 1))
    GX2 = GZ1
    GZ2 = ReLU(convT(GX2, GW2, 1))
    GX3 = GZ2
    GZ3 = ReLU(convT(GX3, GW3, 1))
    GX4 = GZ3
    GZ4 = ReLU(convT(GX4, GW4, 1))
    GX5 = GZ4
    GZ5 = ReLU(convT(GX5, GW5, 1))
    GX6 = GZ5
    GZ6 = tanh(convT(GX6, GW6, 1))
    return GZ6, GX6, GW6, GX5, GW5, GX4, GW4, GX3, GW3, GX2, GW2, GX1, GW1, GX0
