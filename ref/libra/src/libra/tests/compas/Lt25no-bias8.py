
# < 25
assume(0 <= x02 <= 0)
assume(1 <= x03 <= 1)
assume(0 <= x04 <= 0)

x10 = (0.130071)*x00 + (-0.185578)*x01 + (-0.090262)*x02 + (0.315170)*x03 + (-0.735272)*x04 + (-0.011428)*x05 + (0.330910)*x06 + (0.176042)*x07 + (0.136510)*x08 + (0.115256)*x09 + (0.142926)*x010 + (1.312222)*x011 + (1.398417)*x012 + (0.097794)*x013 + (0.050743)*x014 + (-0.001013)*x015 + (-0.371362)*x016 + (0.311483)*x017 + (0.217870)*x018 + (0.064973)
x11 = (0.665130)*x00 + (-0.688634)*x01 + (-0.086382)*x02 + (-0.067920)*x03 + (0.126585)*x04 + (0.143801)*x05 + (0.210060)*x06 + (0.133668)*x07 + (0.297718)*x08 + (-0.104790)*x09 + (0.122913)*x010 + (-0.075650)*x011 + (0.324752)*x012 + (-0.346318)*x013 + (0.356637)*x014 + (-0.215507)*x015 + (-0.343832)*x016 + (-0.542878)*x017 + (0.533857)*x018 + (0.017901)
x12 = (0.453543)*x00 + (0.649689)*x01 + (-0.284835)*x02 + (0.243793)*x03 + (-0.103419)*x04 + (-0.304370)*x05 + (0.603504)*x06 + (0.353062)*x07 + (-0.450164)*x08 + (0.511796)*x09 + (0.512334)*x010 + (-0.373468)*x011 + (-0.648291)*x012 + (-0.365014)*x013 + (0.402769)*x014 + (0.197121)*x015 + (0.474559)*x016 + (0.125797)*x017 + (-0.050088)*x018 + (0.314974)
x13 = (0.094066)*x00 + (0.636761)*x01 + (0.240529)*x02 + (-0.008198)*x03 + (-0.693558)*x04 + (-0.012391)*x05 + (-0.122081)*x06 + (0.074436)*x07 + (-0.046293)*x08 + (0.417012)*x09 + (0.163733)*x010 + (0.476740)*x011 + (0.717425)*x012 + (0.131155)*x013 + (-0.345577)*x014 + (-0.323629)*x015 + (0.203079)*x016 + (0.150590)*x017 + (-0.020016)*x018 + (-0.034496)
x14 = (0.296841)*x00 + (0.423254)*x01 + (-0.180370)*x02 + (0.106583)*x03 + (0.028726)*x04 + (0.072922)*x05 + (0.220163)*x06 + (0.202644)*x07 + (-0.103014)*x08 + (0.220014)*x09 + (0.255302)*x010 + (1.195900)*x011 + (1.296548)*x012 + (-0.358675)*x013 + (0.147305)*x014 + (-0.152720)*x015 + (0.266020)*x016 + (-0.038536)*x017 + (0.265668)*x018 + (-0.046225)
#
ReLU(x10)
ReLU(x11)
ReLU(x12)
ReLU(x13)
ReLU(x14)

x20 = (-0.775483)*x10 + (-0.801272)*x11 + (0.365687)*x12 + (-0.130476)*x13 + (-0.629037)*x14 + (-0.201921)
x21 = (-0.321818)*x10 + (0.079547)*x11 + (-0.210567)*x12 + (-0.692695)*x13 + (0.684614)*x14 + (-0.063498)
x22 = (-0.326404)*x10 + (-0.523765)*x11 + (-0.353471)*x12 + (-0.241756)*x13 + (0.421800)*x14 + (-0.006558)
x23 = (0.904932)*x10 + (0.326379)*x11 + (-0.343922)*x12 + (0.212169)*x13 + (0.885922)*x14 + (-0.111864)
x24 = (-0.137656)*x10 + (0.980538)*x11 + (0.195116)*x12 + (-0.508455)*x13 + (0.064362)*x14 + (0.111274)
#
ReLU(x20)
ReLU(x21)
ReLU(x22)
ReLU(x23)
ReLU(x24)

x30 = (0.149519)*x20 + (0.284042)*x21 + (-0.005432)*x22 + (1.118630)*x23 + (-0.826820)*x24 + (-0.108382)
x31 = (-0.355163)*x20 + (0.333874)*x21 + (-0.428100)*x22 + (-0.690745)*x23 + (0.359286)*x24 + (-0.165285)
x32 = (-0.064078)*x20 + (0.199626)*x21 + (0.690189)*x22 + (0.664587)*x23 + (-0.154061)*x24 + (-0.159461)
x33 = (-0.082376)*x20 + (0.734253)*x21 + (-0.133676)*x22 + (0.578862)*x23 + (0.761623)*x24 + (-0.113656)
x34 = (0.392305)*x20 + (0.348727)*x21 + (0.021167)*x22 + (0.068982)*x23 + (0.005625)*x24 + (0.040421)
#
ReLU(x30)
ReLU(x31)
ReLU(x32)
ReLU(x33)
ReLU(x34)

x40 = (-0.124095)*x30 + (-0.495305)*x31 + (0.420059)*x32 + (-0.488480)*x33 + (-0.508968)*x34 + (0.000000)
x41 = (0.939921)*x30 + (-0.229118)*x31 + (0.736074)*x32 + (0.509484)*x33 + (0.287592)*x34 + (-0.208957)
x42 = (-0.554892)*x30 + (-0.334354)*x31 + (0.149511)*x32 + (0.696537)*x33 + (0.235148)*x34 + (-0.053460)
x43 = (0.346458)*x30 + (-0.129697)*x31 + (0.402363)*x32 + (-0.708402)*x33 + (-0.558229)*x34 + (-0.216753)
x44 = (0.655621)*x30 + (-0.342348)*x31 + (1.254205)*x32 + (0.122046)*x33 + (-0.367836)*x34 + (-0.051283)
#
ReLU(x40)
ReLU(x41)
ReLU(x42)
ReLU(x43)
ReLU(x44)

x50 = (0.429286)*x40 + (0.200846)*x41 + (0.174085)*x42 + (0.751464)*x43 + (-0.894895)*x44 + (0.214546)
x51 = (-0.799044)*x40 + (0.725224)*x41 + (0.741623)*x42 + (0.933728)*x43 + (-0.049770)*x44 + (-0.074353)
x52 = (0.779821)*x40 + (1.157219)*x41 + (-1.064972)*x42 + (-0.703941)*x43 + (0.209264)*x44 + (-0.163179)
#

