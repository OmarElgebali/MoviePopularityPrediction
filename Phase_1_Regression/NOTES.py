"""
<2> Notes:
    <2.1> Length of each vector =>  features_dictionary_list
        (genres:  20),
        (keywords:  7852),
        (production_companies:  3483),
        (production_countries:  65),
        (spoken_languages:  56)

    <2.2> Nulls in columns:
        homepage    (1910)
        overview    (1)
        runtime     (1)
        tagline     (383)

    <2.3> Empty dictionary lists:
        genres :  0
        keywords :  189
        production_companies :  87
        production_countries :  37
        spoken_languages :  10

    <2.4 BONUS> Dataset analysis:
        Cast #(0's):  2
        Cast min  row length:  2
        Cast max  row length:  28704
        Cast mean row length:  3624.1796900758327
        Crew #(0's):  1
        Crew min  row length:  2
        Crew max  row length:  31672
        Crew mean row length:  4470.937026046819

    <2.5 BONUS> Nulls in columns:
        movie_id    0
        title       0
        cast        0
        crew        0

    <2.6> Vector to value:
        1. hashing                                  => Failed - totally random
        2. string to int                            => Failed - huge distances
        3. Jaccard based on fixed-reference [1's]   => Failed - (collision) same inputs perform 1 output
        4. Concentration Point                      => Passed

    <2.7> Title Examples after converting to weighted number
        <EX: 1>
        indices = [12, 1271, 2567]
            Without Limits          12
              (0, 3301)	0.6757595020857902
              (0, 1787)	0.7371221712448791
            [array(['without', 'limits'], dtype='<U14')]
            2512.1228373683075

            Without Men             1271
              (0, 3301)	0.7874454680285349
              (0, 1933)	0.6163843240068015
            [array(['without', 'men'], dtype='<U14')]
            2701.3475847105547

            Without a Paddle        2567
              (0, 3301)	0.6757595020857902
              (0, 2178)	0.7371221712448791
            [array(['without', 'paddle'], dtype='<U14')]
            2716.113570914537

        <EX:2>
        indices = [1267, 1669, 2161]
            City of Angels          1267
              (0, 2131)	0.330643160056703
              (0, 576)	0.6082985764939346
              (0, 155)	0.7215593825480827
            [array(['of', 'city', 'angels'], dtype='<U14')]
            703.692846765208

            City of Ember           1669
              (0, 2131)	0.31012213256472
              (0, 967)	0.7604620261591809
              (0, 576)	0.5705451512925017
            [array(['of', 'ember', 'city'], dtype='<U14')]
            1052.0268998355696

            City of Ghosts          2161
              (0, 2131)	0.32550354292112677
              (0, 1236)	0.7317339033019945
              (0, 576)	0.5988429997121294
            [array(['of', 'ghosts', 'city'], dtype='<U14')]
            1174.2550354358452

        <EX:3>
        indices = [477, 1423]
            Abandon                 477
              (0, 56)	1.0
            [array(['abandon'], dtype='<U14')]
            57.0

            Zulu                    1423
              (0, 3372)	1.0
            [array(['zulu'], dtype='<U14')]
            3373.0

        <EX:4>
        indices = [202, 1170, 1727, 2383]
            The Prince of Tides     202
              (0, 3004)	0.7236550924302424
              (0, 2972)	0.1940615049851207
              (0, 2339)	0.5929353707541964
              (0, 2131)	0.2951119882201852
            [array(['tides', 'the', 'prince', 'of'], dtype='<U14')]
            2640.5311262590976

            The Prince of Egypt     1170
              (0, 2972)	0.19907776539570768
              (0, 2339)	0.6082620488946089
              (0, 2131)	0.30274028412210574
              (0, 947)	0.7062107642688907
            [array(['the', 'prince', 'of', 'egypt'], dtype='<U14')]
            1833.4736740865087

            The Prince & Me         1727
              (0, 2972)	0.24380687012101515
              (0, 2339)	0.7449273205353547
              (0, 1916)	0.6210004003233904
            [array(['the', 'prince', 'me'], dtype='<U14')]
            2272.688642489665

            The Prince              2383
              (0, 2972)	0.3110534356423427
              (0, 2339)	0.9503924243043476
            [array(['the', 'prince'], dtype='<U14')]
            2496.0882087875925

        <EX:5>
        indices = [1210, 1659, 2132]
            A Nightmare on Elm Street                       1210
                (0, 2870) 0.4870210238466554
                (0, 2140) 0.40621913522905945
                (0, 2088) 0.531601735524842
                (0, 963) 0.5614233084688298
            [array(['street', 'on', 'nightmare', 'elm'], dtype='<U14')]
            1973.3923832564785

            A Nightmare on Elm Street 5: The Dream Child    1659
                (0, 2972) 0.1319349107519157
                (0, 2870) 0.3912568123876527
                (0, 2140) 0.3263432094271018
                (0, 2088) 0.42707150270104716
                (0, 963) 0.4510291821422623
                (0, 880) 0.41796321570151174
                (0, 547) 0.4031138232598321
            [array(['the', 'street', 'on', 'nightmare', 'elm', 'dream', 'child'], dtype='<U14')]
            1620.5506995838928

            A Nightmare on Elm Street 5: Dream Warriors     2132
                (0, 3229) 0.44581690132697166
                (0, 2870) 0.3867352859370279
                (0, 2140) 0.3225718515703485
                (0, 2088) 0.4221360867935597
                (0, 963) 0.44581690132697166
                (0, 880) 0.41313305894679647
            [array(['warriors', 'street', 'on', 'nightmare', 'elm', 'dream'], dtype='<U14')]
            2018.0985059532932


        <EX:6>
        indices = [2532, 2915, 607, 501, 2503, 2040, 532, 2352, 1000, 2601, 3025, 864]
            Big                                 2532
                (0, 336) 1.0
            [array(['big'], dtype='<U14')]
            337.0

            Big Daddy                           2915
                (0, 725) 0.7707389178576312
                (0, 336) 0.6371510970716818
            [array(['daddy', 'big'], dtype='<U14')]
            549.9551569137818

            Big Eyes                            607
                (0, 1034) 0.7471662917030272
                (0, 336) 0.6646371435172329
            [array(['eyes', 'big'], dtype='<U14')]
            706.4013334989148

            Big Fat Liar                        501
                (0, 1769) 0.6186621534457344
                (0, 1067) 0.6186621534457344
                (0, 336) 0.4842667444578177
            [array(['liar', 'fat', 'big'], dtype='<U14')]
            1114.64397009807

            Big Fish                            2503
                (0, 1109) 0.7983356385901912
                (0, 336) 0.6022127598754379
            [array(['fish', 'big'], dtype='<U14')]
            777.6227227179701

            Big Hero 6                          2040
                (0, 1411) 0.7874454680285349
                (0, 336) 0.6163843240068015
            [array(['hero', 'big'], dtype='<U14')]
            939.9960917864374

            Big Miracle -> 532
                (0, 1970) 0.7983356385901912
                (0, 336) 0.6022127598754379
            [array(['miracle', 'big'], dtype='<U14')]
            1268.4068938178045

            Big Momma's House                   1000
                (0, 1991) 0.6853145896392402
                (0, 1472) 0.512932141271589
                (0, 336) 0.5169569920971592
            [array(['momma', 'house', 'big'], dtype='<U14')]
            1337.9811285428696

            Big Momma's House 2                 1000
                (0, 1991) 0.6853145896392402
                (0, 1472) 0.512932141271589
                (0, 336) 0.5169569920971592
            [array(['momma', 'house', 'big'], dtype='<U14')]
            1337.9811285428696

            Big Mommas: Like Father, Like Son   2601
                (0, 2772) 0.36557306382694665
                (0, 1992) 0.41077728849682266
                (0, 1781) 0.6847718734723083
                (0, 1072) 0.37658161261648837
                (0, 336) 0.29477470142426937
            [array(['son', 'mommas', 'like', 'father', 'big'], dtype='<U14')]
            1667.5843619432183

            Big Trouble --->3025
                (0, 3079) 0.7874454680285349
                (0, 336) 0.6163843240068015
            [array(['trouble', 'big'], dtype='<U14')]
            1875.6216556001839

            Big Trouble in Little China         864
                (0, 3079) 0.5102421599434531
                (0, 1797) 0.4135241894343486
                (0, 1520) 0.3152064551485881
                (0, 551) 0.5565749466153217
                (0, 336) 0.3993994271424505
            [array(['trouble', 'little', 'in', 'china', 'big'], dtype='<U14')]
            1474.4401929154214

"""
