
# caucasian
assume(0 <= x05 <= 0)
assume(0 <= x06 <= 0)
assume(0 <= x07 <= 0)
assume(0 <= x08 <= 0)
assume(0 <= x09 <= 0)
assume(1 <= x010 <= 1)

x10 = (-0.222990)*x00 + (0.068028)*x01 + (-0.145322)*x02 + (0.306774)*x03 + (-0.642747)*x04 + (0.079049)*x05 + (0.206524)*x06 + (0.071934)*x07 + (-0.124329)*x08 + (-0.423587)*x09 + (-0.441887)*x010 + (0.136059)*x011 + (0.291678)*x012 + (-0.074840)*x013 + (0.339184)*x014 + (-0.204782)*x015 + (0.218028)*x016 + (-0.197815)*x017 + (-0.458457)*x018 + (-0.049497)
x11 = (0.066561)*x00 + (0.131362)*x01 + (0.291804)*x02 + (0.073818)*x03 + (0.337483)*x04 + (0.262900)*x05 + (0.287551)*x06 + (0.109264)*x07 + (0.186348)*x08 + (0.310300)*x09 + (-0.032121)*x010 + (-0.978444)*x011 + (-1.056862)*x012 + (-0.178621)*x013 + (-0.094244)*x014 + (0.019520)*x015 + (0.063613)*x016 + (0.251637)*x017 + (0.238724)*x018 + (0.095588)
x12 = (-0.370465)*x00 + (-0.447610)*x01 + (0.015745)*x02 + (0.075023)*x03 + (-0.315437)*x04 + (-0.471309)*x05 + (0.182406)*x06 + (0.167940)*x07 + (-0.449417)*x08 + (-0.012984)*x09 + (0.232168)*x010 + (0.362500)*x011 + (0.235199)*x012 + (0.000280)*x013 + (0.068103)*x014 + (0.177066)*x015 + (0.132451)*x016 + (0.168460)*x017 + (-0.440822)*x018 + (-0.042026)
x13 = (0.139228)*x00 + (0.154206)*x01 + (0.371642)*x02 + (0.359394)*x03 + (0.034272)*x04 + (0.093403)*x05 + (0.039729)*x06 + (-0.141416)*x07 + (0.035217)*x08 + (0.224718)*x09 + (-0.289621)*x010 + (1.316607)*x011 + (1.093802)*x012 + (0.286373)*x013 + (0.293739)*x014 + (-0.433530)*x015 + (-0.378603)*x016 + (-0.167496)*x017 + (0.038682)*x018 + (-0.021349)
x14 = (0.222560)*x00 + (-0.182833)*x01 + (0.108238)*x02 + (-0.049568)*x03 + (0.013118)*x04 + (-0.038691)*x05 + (-0.246196)*x06 + (0.239124)*x07 + (0.343037)*x08 + (-0.133980)*x09 + (-0.122427)*x010 + (0.489776)*x011 + (0.700191)*x012 + (0.634281)*x013 + (-0.308836)*x014 + (-0.269135)*x015 + (0.365513)*x016 + (-0.119389)*x017 + (0.247150)*x018 + (0.124162)
#
ReLU(x10)
ReLU(x11)
ReLU(x12)
ReLU(x13)
ReLU(x14)

x20 = (-0.319010)*x10 + (0.739952)*x11 + (-0.077187)*x12 + (-0.856063)*x13 + (0.543365)*x14 + (0.100330)
x21 = (0.159774)*x10 + (-0.439186)*x11 + (0.569862)*x12 + (0.154091)*x13 + (0.779119)*x14 + (0.147175)
x22 = (-0.603073)*x10 + (0.823002)*x11 + (-0.391282)*x12 + (-0.689371)*x13 + (0.488681)*x14 + (0.152651)
x23 = (0.053294)*x10 + (-0.323621)*x11 + (-0.819650)*x12 + (0.221279)*x13 + (-0.409515)*x14 + (0.109400)
x24 = (0.580067)*x10 + (0.210898)*x11 + (0.110382)*x12 + (-0.669025)*x13 + (0.142563)*x14 + (0.037431)
#
ReLU(x20)
ReLU(x21)
ReLU(x22)
ReLU(x23)
ReLU(x24)

x30 = (0.099250)*x20 + (-0.080838)*x21 + (-0.124413)*x22 + (0.282323)*x23 + (0.635700)*x24 + (-0.028692)
x31 = (-0.716570)*x20 + (-0.383483)*x21 + (0.138335)*x22 + (0.904828)*x23 + (-0.205160)*x24 + (0.194227)
x32 = (0.669538)*x20 + (0.242708)*x21 + (-0.431364)*x22 + (0.199288)*x23 + (0.078306)*x24 + (-0.134505)
x33 = (-0.483197)*x20 + (0.776679)*x21 + (-0.571685)*x22 + (-0.053129)*x23 + (-0.688561)*x24 + (0.416840)
x34 = (0.360487)*x20 + (-0.604544)*x21 + (0.582847)*x22 + (-0.409150)*x23 + (-0.159258)*x24 + (0.154526)
#
ReLU(x30)
ReLU(x31)
ReLU(x32)
ReLU(x33)
ReLU(x34)

x40 = (-0.018786)*x30 + (-0.547626)*x31 + (0.698414)*x32 + (0.236493)*x33 + (-0.135352)*x34 + (0.007493)
x41 = (0.917363)*x30 + (0.537254)*x31 + (-0.398699)*x32 + (-0.684900)*x33 + (-0.342654)*x34 + (0.204202)
x42 = (-0.805804)*x30 + (0.505075)*x31 + (-0.172856)*x32 + (0.460443)*x33 + (-1.018691)*x34 + (0.274323)
x43 = (-0.006239)*x30 + (-0.362454)*x31 + (-0.210681)*x32 + (-0.297744)*x33 + (0.771760)*x34 + (0.155052)
x44 = (-0.380228)*x30 + (0.889285)*x31 + (0.238377)*x32 + (0.865975)*x33 + (-0.383214)*x34 + (0.303543)
#
ReLU(x40)
ReLU(x41)
ReLU(x42)
ReLU(x43)
ReLU(x44)

x50 = (0.277994)*x40 + (0.176052)*x41 + (-1.103328)*x42 + (0.246745)*x43 + (-1.119145)*x44 + (-0.242211)
x51 = (-0.484712)*x40 + (-0.399924)*x41 + (-0.731547)*x42 + (-0.559929)*x43 + (0.782240)*x44 + (0.034138)
x52 = (0.614073)*x40 + (-1.124734)*x41 + (0.934188)*x42 + (-1.068694)*x43 + (0.553851)*x44 + (0.144742)
#

