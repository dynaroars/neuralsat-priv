
# caucasian
assume(1 <= x05 <= 1)
assume(0 <= x06 <= 0)
assume(0 <= x07 <= 0)
assume(0 <= x08 <= 0)
assume(0 <= x09 <= 0)
assume(0 <= x010 <= 0)

x10 = (-0.041120)*x00 + (0.046984)*x01 + (-0.120198)*x02 + (0.068320)*x03 + (0.041289)*x04 + (-0.221826)*x05 + (-0.227039)*x06 + (0.392076)*x07 + (-0.195087)*x08 + (0.148283)*x09 + (-0.249805)*x010 + (0.885980)*x011 + (1.325040)*x012 + (0.330533)*x013 + (-0.066403)*x014 + (-0.173214)*x015 + (0.323554)*x016 + (0.399193)*x017 + (0.158498)*x018 + (0.206448)
x11 = (-0.261950)*x00 + (-0.470653)*x01 + (-0.337756)*x02 + (0.085224)*x03 + (-0.196721)*x04 + (-0.104744)*x05 + (0.222930)*x06 + (-0.312856)*x07 + (0.276226)*x08 + (-0.382958)*x09 + (0.023162)*x010 + (0.160749)*x011 + (0.124937)*x012 + (-0.098887)*x013 + (0.025816)*x014 + (-0.249010)*x015 + (-0.383969)*x016 + (0.027206)*x017 + (-0.201688)*x018 + (0.000000)
x12 = (0.397418)*x00 + (-0.408829)*x01 + (-0.306973)*x02 + (0.545842)*x03 + (-0.593998)*x04 + (0.007380)*x05 + (-0.356873)*x06 + (-0.385586)*x07 + (-0.247424)*x08 + (-0.481069)*x09 + (-0.357282)*x010 + (0.357911)*x011 + (-0.149925)*x012 + (-0.047656)*x013 + (0.025980)*x014 + (0.057178)*x015 + (0.276636)*x016 + (-0.339671)*x017 + (0.129575)*x018 + (-0.094179)
x13 = (-0.022578)*x00 + (0.296249)*x01 + (0.546729)*x02 + (-0.193788)*x03 + (0.335249)*x04 + (-0.152612)*x05 + (-0.168081)*x06 + (-0.109009)*x07 + (0.568227)*x08 + (0.420403)*x09 + (0.533152)*x010 + (0.968059)*x011 + (0.966226)*x012 + (-0.007946)*x013 + (-0.043597)*x014 + (-0.068630)*x015 + (-0.313270)*x016 + (0.000648)*x017 + (-0.146549)*x018 + (0.111738)
x14 = (-0.061200)*x00 + (0.171422)*x01 + (0.182554)*x02 + (0.507769)*x03 + (-0.176372)*x04 + (-0.313051)*x05 + (-0.307171)*x06 + (-0.117332)*x07 + (0.195258)*x08 + (0.231246)*x09 + (0.150868)*x010 + (-1.090319)*x011 + (-1.006666)*x012 + (0.417004)*x013 + (0.249938)*x014 + (0.398936)*x015 + (0.473316)*x016 + (0.174484)*x017 + (-0.127068)*x018 + (0.078536)
#
ReLU(x10)
ReLU(x11)
ReLU(x12)
ReLU(x13)
ReLU(x14)

x20 = (0.159798)*x10 + (0.479188)*x11 + (0.347497)*x12 + (0.365969)*x13 + (-0.608089)*x14 + (0.232214)
x21 = (0.886408)*x10 + (-0.366068)*x11 + (0.069794)*x12 + (0.749410)*x13 + (-0.522638)*x14 + (0.026923)
x22 = (0.030413)*x10 + (0.273026)*x11 + (0.056602)*x12 + (0.358894)*x13 + (-0.641248)*x14 + (-0.041607)
x23 = (0.654688)*x10 + (0.417667)*x11 + (-0.320094)*x12 + (0.197412)*x13 + (0.549683)*x14 + (0.201033)
x24 = (0.071179)*x10 + (-0.405307)*x11 + (-0.251975)*x12 + (-0.317704)*x13 + (-0.530664)*x14 + (-0.102981)
#
ReLU(x20)
ReLU(x21)
ReLU(x22)
ReLU(x23)
ReLU(x24)

x30 = (-0.671450)*x20 + (0.454771)*x21 + (-0.535458)*x22 + (-0.302924)*x23 + (-0.202255)*x24 + (-0.078880)
x31 = (0.978207)*x20 + (-0.272593)*x21 + (0.364593)*x22 + (-0.528412)*x23 + (-0.542455)*x24 + (0.045071)
x32 = (-0.880964)*x20 + (-0.372518)*x21 + (-0.575578)*x22 + (0.396146)*x23 + (0.394536)*x24 + (0.248216)
x33 = (0.330858)*x20 + (1.054669)*x21 + (-0.528802)*x22 + (0.187052)*x23 + (-0.692853)*x24 + (0.165659)
x34 = (-0.532680)*x20 + (-0.552770)*x21 + (0.630747)*x22 + (0.609730)*x23 + (-0.259575)*x24 + (0.115832)
#
ReLU(x30)
ReLU(x31)
ReLU(x32)
ReLU(x33)
ReLU(x34)

x40 = (0.721880)*x30 + (-0.118095)*x31 + (0.089450)*x32 + (-0.692325)*x33 + (0.143469)*x34 + (-0.020779)
x41 = (0.452275)*x30 + (-0.365087)*x31 + (-0.383337)*x32 + (-0.458649)*x33 + (-0.673774)*x34 + (0.000000)
x42 = (-0.199018)*x30 + (0.138397)*x31 + (-0.436595)*x32 + (0.080163)*x33 + (0.257645)*x34 + (0.144381)
x43 = (0.175100)*x30 + (0.637440)*x31 + (-0.884043)*x32 + (0.828305)*x33 + (-1.214658)*x34 + (0.228664)
x44 = (-0.867832)*x30 + (-0.718006)*x31 + (-0.466475)*x32 + (0.102836)*x33 + (-0.900882)*x34 + (-0.367107)
#
ReLU(x40)
ReLU(x41)
ReLU(x42)
ReLU(x43)
ReLU(x44)

x50 = (0.185463)*x40 + (0.086534)*x41 + (-0.176875)*x42 + (-1.195729)*x43 + (-0.520928)*x44 + (0.147567)
x51 = (-0.611056)*x40 + (0.436778)*x41 + (0.902691)*x42 + (-0.329941)*x43 + (-0.763830)*x44 + (0.015474)
x52 = (-0.119237)*x40 + (-0.507699)*x41 + (-0.439990)*x42 + (0.606342)*x43 + (-0.312236)*x44 + (-0.122606)
#

