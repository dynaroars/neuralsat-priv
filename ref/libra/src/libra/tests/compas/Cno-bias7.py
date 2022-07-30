
# caucasian
assume(0 <= x05 <= 0)
assume(0 <= x06 <= 0)
assume(0 <= x07 <= 0)
assume(0 <= x08 <= 0)
assume(0 <= x09 <= 0)
assume(1 <= x010 <= 1)

x10 = (-0.141791)*x00 + (0.400445)*x01 + (-0.188714)*x02 + (-0.408997)*x03 + (-0.113025)*x04 + (-0.007386)*x05 + (-0.076653)*x06 + (0.063539)*x07 + (-0.195587)*x08 + (-0.098265)*x09 + (0.043043)*x010 + (-0.932677)*x011 + (-0.863383)*x012 + (0.067953)*x013 + (-0.141872)*x014 + (0.169876)*x015 + (0.366996)*x016 + (0.368870)*x017 + (0.035812)*x018 + (0.131423)
x11 = (0.391304)*x00 + (-0.235139)*x01 + (-0.393648)*x02 + (0.002031)*x03 + (0.073365)*x04 + (-0.304461)*x05 + (0.167287)*x06 + (-0.126939)*x07 + (0.226297)*x08 + (0.398464)*x09 + (0.313025)*x010 + (-0.033849)*x011 + (-0.088191)*x012 + (0.073148)*x013 + (-0.491679)*x014 + (0.318372)*x015 + (-0.323442)*x016 + (-0.483339)*x017 + (-0.535462)*x018 + (-0.063654)
x12 = (-0.556026)*x00 + (-0.124515)*x01 + (0.519842)*x02 + (0.256853)*x03 + (-0.559318)*x04 + (-0.203042)*x05 + (-0.289104)*x06 + (-0.085706)*x07 + (-0.289614)*x08 + (-0.328656)*x09 + (-0.110579)*x010 + (0.527819)*x011 + (0.276538)*x012 + (0.437719)*x013 + (0.092088)*x014 + (-0.070146)*x015 + (-0.023860)*x016 + (0.438948)*x017 + (0.415309)*x018 + (0.103305)
x13 = (0.008490)*x00 + (0.396846)*x01 + (0.350667)*x02 + (0.443573)*x03 + (0.086490)*x04 + (0.150864)*x05 + (0.139769)*x06 + (0.232126)*x07 + (0.033195)*x08 + (0.126543)*x09 + (0.207077)*x010 + (0.921340)*x011 + (1.395236)*x012 + (-0.220156)*x013 + (-0.400797)*x014 + (0.077924)*x015 + (0.305696)*x016 + (-0.077830)*x017 + (-0.269599)*x018 + (0.040464)
x14 = (-0.280514)*x00 + (0.131278)*x01 + (-0.310522)*x02 + (0.224578)*x03 + (0.376081)*x04 + (0.217551)*x05 + (-0.373302)*x06 + (0.012671)*x07 + (-0.384050)*x08 + (-0.429183)*x09 + (-0.105890)*x010 + (-0.197358)*x011 + (0.299067)*x012 + (0.141318)*x013 + (-0.131552)*x014 + (-0.446571)*x015 + (-0.394874)*x016 + (-0.456775)*x017 + (0.350672)*x018 + (-0.027656)
#
ReLU(x10)
ReLU(x11)
ReLU(x12)
ReLU(x13)
ReLU(x14)

x20 = (-0.429551)*x10 + (0.777859)*x11 + (0.028237)*x12 + (0.338484)*x13 + (0.667085)*x14 + (-0.019975)
x21 = (-0.102672)*x10 + (0.069403)*x11 + (0.663826)*x12 + (-0.811961)*x13 + (-0.174637)*x14 + (-0.001649)
x22 = (-0.666517)*x10 + (0.235837)*x11 + (0.135207)*x12 + (0.790035)*x13 + (-0.019277)*x14 + (-0.061937)
x23 = (-0.210362)*x10 + (-0.235873)*x11 + (-0.320187)*x12 + (0.447062)*x13 + (-0.131515)*x14 + (0.093816)
x24 = (-0.177899)*x10 + (0.360087)*x11 + (-0.193996)*x12 + (-0.396206)*x13 + (-0.271075)*x14 + (0.041416)
#
ReLU(x20)
ReLU(x21)
ReLU(x22)
ReLU(x23)
ReLU(x24)

x30 = (-0.379268)*x20 + (-0.637650)*x21 + (-0.371154)*x22 + (0.365003)*x23 + (-0.193537)*x24 + (-0.037872)
x31 = (-0.445553)*x20 + (-0.452148)*x21 + (-0.352962)*x22 + (0.318492)*x23 + (-0.405771)*x24 + (0.293813)
x32 = (-0.656098)*x20 + (-0.392112)*x21 + (-0.272967)*x22 + (-0.560409)*x23 + (0.187250)*x24 + (0.148315)
x33 = (0.428110)*x20 + (0.667354)*x21 + (0.699775)*x22 + (-0.499636)*x23 + (0.370837)*x24 + (-0.066333)
x34 = (0.713687)*x20 + (0.599077)*x21 + (0.912445)*x22 + (0.686982)*x23 + (-0.306952)*x24 + (-0.015507)
#
ReLU(x30)
ReLU(x31)
ReLU(x32)
ReLU(x33)
ReLU(x34)

x40 = (0.597970)*x30 + (1.216013)*x31 + (0.246252)*x32 + (-0.188928)*x33 + (-0.345490)*x34 + (0.354272)
x41 = (-0.004352)*x30 + (-0.825291)*x31 + (0.369324)*x32 + (0.871024)*x33 + (0.288666)*x34 + (0.019800)
x42 = (0.724607)*x30 + (0.085073)*x31 + (-0.997257)*x32 + (-0.207698)*x33 + (0.770431)*x34 + (0.013581)
x43 = (0.170470)*x30 + (0.245030)*x31 + (-0.323962)*x32 + (0.221598)*x33 + (0.898004)*x34 + (0.006403)
x44 = (0.362694)*x30 + (-0.097787)*x31 + (-0.471650)*x32 + (0.549003)*x33 + (-0.237213)*x34 + (-0.155220)
#
ReLU(x40)
ReLU(x41)
ReLU(x42)
ReLU(x43)
ReLU(x44)

x50 = (0.954096)*x40 + (-0.388557)*x41 + (-0.921098)*x42 + (-0.678123)*x43 + (0.416431)*x44 + (-0.003982)
x51 = (0.171906)*x40 + (-0.802512)*x41 + (0.804712)*x42 + (0.216768)*x43 + (0.406426)*x44 + (-0.050132)
x52 = (-0.318169)*x40 + (1.105078)*x41 + (0.064283)*x42 + (0.354296)*x43 + (-0.475470)*x44 + (0.039111)
#

