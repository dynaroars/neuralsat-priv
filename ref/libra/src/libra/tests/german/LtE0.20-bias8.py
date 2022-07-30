
assume (x05 <= 0.04)

x10 = (-0.263086)*x00 + (0.331205)*x01 + (0.713287)*x02 + (0.538914)*x03 + (0.068219)*x04 + (-0.411893)*x05 + (-0.265148)*x06 + (0.384549)*x07 + (0.374160)*x08 + (-0.058207)*x09 + (-0.431131)*x010 + (0.296419)*x011 + (-0.008834)*x012 + (-0.079324)*x013 + (0.171519)*x014 + (0.009140)*x015 + (-0.552380)*x016 + (0.037443)
x11 = (0.446398)*x00 + (0.010114)*x01 + (0.007105)*x02 + (0.232848)*x03 + (-0.070799)*x04 + (-0.112862)*x05 + (0.039272)*x06 + (0.442880)*x07 + (0.043568)*x08 + (0.204306)*x09 + (-0.431398)*x010 + (-0.577295)*x011 + (-0.142872)*x012 + (0.154890)*x013 + (-0.289605)*x014 + (0.478886)*x015 + (0.097373)*x016 + (0.110291)
x12 = (0.105305)*x00 + (0.729090)*x01 + (-0.451742)*x02 + (0.181870)*x03 + (-0.040363)*x04 + (0.490276)*x05 + (0.348071)*x06 + (0.025922)*x07 + (-0.504722)*x08 + (0.335952)*x09 + (-0.226319)*x010 + (-0.092264)*x011 + (0.494220)*x012 + (-0.058019)*x013 + (-0.015573)*x014 + (-0.182882)*x015 + (-0.301178)*x016 + (0.046724)
x13 = (0.108205)*x00 + (-0.429515)*x01 + (-0.288302)*x02 + (-0.314552)*x03 + (-0.280338)*x04 + (0.065574)*x05 + (-0.331848)*x06 + (0.215101)*x07 + (-0.341711)*x08 + (0.106427)*x09 + (0.099383)*x010 + (0.388357)*x011 + (-0.086972)*x012 + (-0.404907)*x013 + (0.080249)*x014 + (0.203321)*x015 + (-0.463792)*x016 + (-0.015770)
x14 = (-0.053190)*x00 + (-0.126152)*x01 + (0.179434)*x02 + (-0.012083)*x03 + (0.488933)*x04 + (0.521338)*x05 + (-0.189869)*x06 + (0.057475)*x07 + (-0.181387)*x08 + (-0.623679)*x09 + (-0.274951)*x010 + (0.172782)*x011 + (0.491100)*x012 + (0.321879)*x013 + (-0.027858)*x014 + (0.318645)*x015 + (0.041418)*x016 + (0.000558)
#
ReLU(x10)
ReLU(x11)
ReLU(x12)
ReLU(x13)
ReLU(x14)

x20 = (0.403236)*x10 + (-0.200665)*x11 + (0.337777)*x12 + (-0.137098)*x13 + (-0.388805)*x14 + (-0.047028)
x21 = (0.583811)*x10 + (0.538665)*x11 + (0.483590)*x12 + (0.604192)*x13 + (-0.664957)*x14 + (0.101648)
x22 = (-0.607572)*x10 + (-0.562025)*x11 + (-0.343085)*x12 + (0.665043)*x13 + (-0.207029)*x14 + (0.000000)
x23 = (-0.424595)*x10 + (-0.299144)*x11 + (0.588098)*x12 + (0.347607)*x13 + (0.356813)*x14 + (0.060837)
x24 = (0.275833)*x10 + (-0.137451)*x11 + (-0.496413)*x12 + (-0.396748)*x13 + (0.486034)*x14 + (-0.046508)
#
ReLU(x20)
ReLU(x21)
ReLU(x22)
ReLU(x23)
ReLU(x24)

x30 = (-0.716455)*x20 + (-0.670771)*x21 + (-0.550788)*x22 + (-0.519606)*x23 + (-0.102276)*x24 + (0.000000)
x31 = (-0.273494)*x20 + (0.318506)*x21 + (0.175381)*x22 + (0.647897)*x23 + (-0.095257)*x24 + (-0.129309)
x32 = (0.575790)*x20 + (-0.309761)*x21 + (0.729416)*x22 + (0.334414)*x23 + (-0.172993)*x24 + (0.076017)
x33 = (0.378070)*x20 + (-0.620091)*x21 + (-0.545764)*x22 + (-0.191179)*x23 + (0.278028)*x24 + (0.022348)
x34 = (0.324772)*x20 + (-0.350498)*x21 + (-0.603067)*x22 + (0.726424)*x23 + (-0.120804)*x24 + (0.042755)
#
ReLU(x30)
ReLU(x31)
ReLU(x32)
ReLU(x33)
ReLU(x34)

x40 = (0.359440)*x30 + (-0.115241)*x31 + (0.820804)*x32 + (-0.387919)*x33 + (0.534936)*x34 + (0.002816)
x41 = (-0.571904)*x30 + (0.181015)*x31 + (0.322910)*x32 + (0.117905)*x33 + (-0.802888)*x34 + (-0.123400)
x42 = (0.497642)*x30 + (-0.585630)*x31 + (0.857556)*x32 + (-0.274536)*x33 + (0.220633)*x34 + (0.048707)
x43 = (-0.428427)*x30 + (-0.506106)*x31 + (-0.713456)*x32 + (0.572008)*x33 + (-0.361882)*x34 + (0.101862)
x44 = (-0.449407)*x30 + (-0.744960)*x31 + (0.340888)*x32 + (-0.518701)*x33 + (-0.170068)*x34 + (-0.014610)
#
ReLU(x40)
ReLU(x41)
ReLU(x42)
ReLU(x43)
ReLU(x44)

x50 = (1.020939)*x40 + (0.355100)*x41 + (-0.291814)*x42 + (-0.822557)*x43 + (0.568855)*x44 + (-0.363629)
x51 = (-0.769847)*x40 + (-0.710463)*x41 + (-0.606786)*x42 + (0.084129)*x43 + (0.783014)*x44 + (0.363629)
