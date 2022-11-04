//
//  main.cpp
//  
//
//  Created by Terry Kong on 3/6/15.
//
//

#include <stdio.h>
#include "PSOLA.h"
#include <iostream>
#include <math.h>

#define DEFAULT_BUFFER_SIZE 512
#define FIXED_BITS        16
#define FIXED_WBITS       0
#define FIXED_FBITS       15
#define Q15_RESOLUTION   (1 << (FIXED_FBITS - 1))
#define LARGEST_Q15_NUM   32767

using namespace std;

int windowLen = 512;
int data[1024] = {-2126,-2096,-2041,-2274,-2257,-2368,-2315,-2330,-2266,-2251,-2205,-2158,
    -2098,-2026,-1909,-1768,-1637,-1510,-1380,-1254,-1137,-1037,-950,-861,
    -841,-886,-937,-1101,-1279,-1443,-1671,-1810,-1829,-1837,-1758,-1595,
    -1466,-1331,-1206,-1136,-1089,-1095,-1135,-1149,-1175,-1210,-1195,-1155,
    -1097,-987,-885,-776,-609,-491,-387,-277,-202,-127,-53,3,
    47,130,219,317,448,559,671,786,827,822,824,780,
    711,647,563,513,469,424,374,299,209,96,-42,-227,
    -441,-682,-957,-1215,-1462,-1655,-1816,-1911,-1942,-1914,-1887,-1893,
    -1941,-2074,-2224,-2432,-2651,-2852,-2874,-2776,-2748,-2526,-2255,-2098,
    -1922,-1803,-1647,-1485,-1307,-1089,-880,-670,-498,-385,-326,-331,
    -338,-337,-336,-303,-249,-166,-75,11,109,226,360,516,
    684,871,1035,1153,1306,1437,1527,1629,1787,1902,2008,2108,
    2149,2152,2086,1990,1846,1715,1588,1469,1382,1304,1247,1198,
    1144,1112,1079,1027,961,864,757,644,494,395,288,162,
    64,-70,-218,-405,-646,-900,-1178,-1444,-1664,-1887,-2089,-2237,
    -2390,-2506,-2572,-2613,-2591,-2540,-2455,-2354,-2244,-2121,-2005,-1885,
    -1790,-1721,-1643,-1594,-1587,-1622,-1679,-1745,-1822,-1873,-1903,-1903,
    -1839,-1771,-1697,-1618,-1569,-1519,-1479,-1421,-1362,-1320,-1284,-1267,
    -1266,-1298,-1331,-1346,-1354,-1321,-1280,-1233,-1190,-1199,-1251,-1327,
    -1418,-1496,-1527,-1504,-1439,-1317,-1163,-1012,-855,-739,-630,-537,
    -430,-308,-204,-84,31,156,274,382,486,563,634,688,
    725,744,748,755,755,738,682,588,450,277,71,-140,
    -331,-473,-555,-601,-643,-709,-816,-992,-1221,-1495,-1764,-2019,
    -2221,-2379,-2490,-2562,-2608,-2597,-2565,-2489,-2384,-2257,-2119,-2010,
    -1935,-1914,-1936,-1959,-1962,-1914,-1805,-1636,-1436,-1228,-1027,-863,
    -721,-587,-443,-262,-55,166,389,587,740,843,918,968,
    1016,1074,1143,1221,1281,1325,1349,1339,1325,1318,1326,1363,
    1417,1478,1533,1576,1602,1595,1570,1530,1482,1432,1372,1308,
    1241,1177,1114,1053,998,937,873,806,721,624,518,396,
    262,115,-51,-231,-427,-629,-822,-1002,-1150,-1259,-1338,-1395,
    -1439,-1487,-1542,-1604,-1665,-1714,-1745,-1754,-1748,-1741,-1745,-1772,
    -1824,-1895,-1971,-2033,-2063,-2049,-1993,-1897,-1778,-1651,-1526,-1416,
    -1324,-1255,-1207,-1172,-1153,-1143,-1139,-1141,-1145,-1150,-1155,-1157,
    -1157,-1156,-1156,-1161,-1175,-1202,-1246,-1302,-1367,-1427,-1471,-1485,
    -1461,-1400,-1305,-1192,-1073,-961,-865,-786,-720,-664,-612,-560,
    -506,-447,-382,-304,-208,-90,48,196,341,467,566,634,
    682,724,779,854,946,1037,1104,1122,1078,978,840,698,
    584,520,510,538,578,599,581,518,418,299,178,62,
    -52,-181,-344,-551,-799,-1073,-1342,-1577,-1756,-1871,-1934,-1970,
    -2009,-2072,-2172,-2301,-2439,-2560,-2636,-2649,-2590,-2463,-2282,-2062,
    -1824,-1579,-1342,-1121,-917,-730,-558,-395,-235,-70,100,275,
    448,608,744,852,929,985,1031,1081,1149,1238,1345,1461,
    1571,1664,1735,1788,1832,1879,1937,2005,2076,2131,2148,2109,
    2006,1835,1612,1357,1096,855,652,500,404,360,358,382,
    414,432,419,365,270,144,6,-123,-223,-289,-326,-358,
    -411,-512,-675,-894,-1147,-1401,-1619,-1781,-1882,-1935,-1967,-2003,
    -2059,-2133,-2203,-2240,-2215,-2117,-1953,-1754,-1561,-1414,-1339,-1343,
    -1411,-1516,-1626,-1715,-1773,-1804,-1814,-1815,-1815,-1811,-1798,-1768,
    -1721,-1661,-1599,-1541,-1494,-1449,-1396,-1315,-1195,-1034,-845,-650,
    -479,-357,-295,-290,-320,-359,-383,-373,-328,-258,-181,-116,
    -74,-59,-63,-73,-75,-60,-25,29,95,170,254,349,
    460,588,731,878,1013,1120,1183,1195,1157,1078,973,858,
    745,641,544,450,351,242,122,-9,-143,-275,-398,-509,
    -604,-683,-748,-800,-841,-878,-918,-972,-1052,-1165,-1317,-1501,
    -1707,-1912,-2093,-2230,-2305,-2311,-2251,-2141,-1996,-1840,-1689,-1553,
    -1438,-1342,-1255,-1165,-1060,-933,-778,-599,-407,-217,-45,93,
    191,251,281,297,317,356,419,507,613,726,838,946,
    1052,1162,1279,1406,1533,1645,1729,1770,1760,1709,1632,1550,
    1488,1464,1482,1542,1629,1726,1816,1886,1929,1945,1933,1896,
    1835,1752,1651,1532,1404,1271,1134,993,842,670,467,227,
    -49,-349,-655,-941,-1186,-1377,-1509,-1589,-1629,-1642,-1639,-1620,
    -1583,-1526,-1447,-1353,-1262,-1196,-1174,-1210,-1307,-1457,-1639,-1831,
    -2015,-2176,-2309,-2415,-2498,-2558,-2593,-2598,-2567,-2494,-2380,-2228,
    -2045,-1839,-1613,-1373,-1129,-892,-681,-516,-415,-390,-439,-545,
    -681,-815,-920,-985,-1010,-1011,-1004,-1004,-1014,-1024,-1020,-983,
    -906,-793,-656,-520,-403,-316,-261,-229,-202,-167,-120,-58,
    11,79,142,198,250,305,364,423,480,528,561,571,
    565,542,505,457,400,335,266,199,140,94,66,44,
    15,-42,-144,-302,-515,-770,-1045,-1313,-1552,-1748,-1897,-2007,
    -2087,-2148,-2194,-2224,-2235,-2219,-2170,-2083,-1958,-1799,-1613,-1414,
    -1216,-1034,-882,-769,-697,-654,-625,-587,-519,-411,-262,-80,
    113,298,458,589,696,792,893,1007,1134,1264,1382,1476,
    1538,1570,1585,1605,1641,1701,1784,1879,1969,2038,2076,2084,
    2071,2051,2039,2044,2070,2117,2176,2240,2297,2336,2352,2332,
    2269,2160,2005,1811,1592,1369,1159,973,821,697,589,482,
    362,230,90,-46,-163,-254,-319,-365,-408,-465,-541,-640,
    -750,-862,-967,-1062,-1156,-1262,-1394,-1559,-1753,-1957,-2150,-2304,
    -2397,-2422,-2382,-2284,-2145,-1977,-1788,-1585,-1374,-1166,-973,-810,
    -693,-628,-614,-638,-680,-719,-740,-738,-715,-684,-661,-651,
    -658,-676,-696,-707,-706,-693,-672,-655,-654,-671,-710,-763,
    -824,-881,-927,-956,-966,-958,-937,-902,-858,-802,-731,-645,
    -540,-418,-285,-150,-21,93,190,275,357,447,554,679,
    814,943,1049,1110,1117,1069,971,836,674,499,315,131,
    -47,-207,-334,-418,-455,-447,-413,-380,-376,-420,-519,-669,
    -851,-1043,-1228,-1401,-1564,-1727,-1893,-2062,-2220,-2347,-2424,-2434,
    -2376,-2263,-2113,-1942};
int data2[1024] = {-1768,-1594,-1417,-1233,-1038,-838,-647,-481,-351,-260,-201,-156,
    -106,-36,59,182,325,479,642,815,1000,1197,1401,1595,
    1761,1881,1943,1953,1926,1888,1868,1883,1939,2027,2125,2206,
    2246,2233,2165,2053,1918,1779,1653,1550,1473,1412,1353,1286,
    1193,1077,944,808,688,602,558,551,563,568,539,459,
    325,150,-45,-230,-389,-517,-622,-727,-853,-1014,-1205,-1410,
    -1602,-1754,-1844,-1869,-1840,-1776,-1696,-1619,-1549,-1485,-1423,-1354,
    -1283,-1219,-1174,-1163,-1194,-1262,-1356,-1460,-1553,-1620,-1651,-1645,
    -1606,-1545,-1471,-1392,-1316,-1245,-1181,-1122,-1067,-1012,-954,-893,
    -829,-765,-704,-653,-614,-593,-587,-595,-617,-650,-695,-750,
    -810,-870,-919,-946,-936,-882,-781,-635,-457,-258,-53,144,
    324,481,612,715,791,842,872,886,887,883,876,870,
    865,863,862,862,866,872,878,880,870,841,784,698,
    588,463,336,218,114,23,-71,-180,-320,-501,-723,-975,
    -1239,-1493,-1721,-1910,-2054,-2161,-2233,-2281,-2312,-2325,-2322,-2302,
    -2259,-2191,-2096,-1973,-1824,-1647,-1452,-1246,-1037,-842,-671,-534,
    -432,-367,-330,-307,-285,-250,-189,-95,35,198,386,588,
    792,987,1165,1323,1461,1584,1698,1805,1906,1997,2072,2125,
    2156,2166,2161,2153,2150,2158,2172,2188,2195,2182,2146,2089,
    2019,1944,1877,1819,1767,1713,1648,1562,1448,1308,1147,971,
    788,604,422,244,72,-95,-257,-417,-578,-743,-914,-1089,
    -1259,-1416,-1541,-1625,-1660,-1647,-1597,-1528,-1460,-1410,-1388,-1391,
    -1414,-1441,-1458,-1455,-1425,-1372,-1301,-1225,-1156,-1101,-1068,-1057,
    -1063,-1083,-1108,-1125,-1129,-1118,-1094,-1065,-1045,-1043,-1067,-1117,
    -1184,-1253,-1309,-1337,-1329,-1283,-1208,-1113,-1008,-899,-792,-686,
    -585,-491,-413,-359,-339,-356,-404,-473,-549,-612,-649,-651,
    -620,-556,-468,-361,-236,-94,65,242,428,615,790,942,
    1066,1159,1220,1253,1258,1232,1176,1084,964,828,691,572,
    484,430,400,370,318,220,72,-121,-336,-540,-710,-831,
    -907,-951,-989,-1044,-1126,-1237,-1362,-1486,-1592,-1672,-1729,-1770,
    -1803,-1834,-1858,-1869,-1856,-1817,-1756,-1688,-1633,-1603,-1602,-1619,
    -1627,-1595,-1497,-1320,-1070,-770,-455,-158,96,304,474,623,
    774,939,1117,1295,1456,1582,1665,1707,1719,1717,1720,1736,
    1767,1811,1868,1936,2016,2112,2225,2345,2456,2537,2569,2544,
    2464,2346,2213,2086,1975,1879,1787,1678,1546,1392,1230,1084,
    967,884,819,748,642,484,279,53,-155,-307,-381,-387,
    -350,-320,-338,-424,-581,-783,-992,-1178,-1328,-1441,-1535,-1624,
    -1716,-1804,-1868,-1887,-1852,-1774,-1680,-1606,-1579,-1610,-1683,-1763,
    -1811,-1797,-1711,-1570,-1407,-1258,-1151,-1092,-1070,-1063,-1044,-1000,
    -929,-841,-745,-644,-536,-409,-252,-69,132,320,463,531,
    513,415,262,87,-75,-202,-283,-325,-339,-339,-335,-326,
    -308,-272,-207,-111,19,179,360,549,733,896,1028,1118,
    1171,1194,1198,1197,1196,1192,1174,1127,1034,895,715,513,
    317,150,29,-45,-80,-92,-95,-98,-99,-90,-70,-40,
    -12,-9,-57,-172,-362,-615,-909,-1214,-1500,-1748,-1947,-2099,
    -2208,-2284,-2332,-2350,-2340,-2299,-2230,-2138,-2032,-1919,-1804,-1683,
    -1552,-1405,-1236,-1050,-855,-668,-504,-379,-303,-273,-275,-290,
    -291,-257,-168,-22,171,394,622,830,1007,1149,1265,1369,
    1477,1597,1725,1846,1940,1991,1990,1943,1867,1788,1731,1711,
    1735,1788,1862,1936,2004,2064,2119,2171,2219,2257,2269,2244,
    2179,2075,1947,1813,1689,1579,1480,1375,1244,1071,855,610,
    359,130,-57,-193,-284,-345,-394,-447,-508,-572,-633,-685,
    -731,-782,-856,-969,-1126,-1324,-1544,-1755,-1931,-2050,-2104,-2100,
    -2054,-1987,-1917,-1852,-1795,-1740,-1680,-1614,-1542,-1468,-1402,-1347,
    -1304,-1263,-1215,-1154,-1074,-984,-898,-832,-803,-820,-877,-953,
    -1024,-1058,-1034,-941,-785,-594,-397,-226,-108,-49,-42,-65,
    -86,-81,-31,65,190,326,443,520,550,536,494,445,
    409,399,415,453,504,560,619,684,765,866,990,1125,
    1251,1342,1375,1331,1211,1022,785,528,274,38,-167,-340,
    -483,-597,-687,-752,-794,-821,-846,-879,-933,-1015,-1122,-1245,
    -1369,-1481,-1574,-1648,-1706,-1756,-1803,-1844,-1869,-1867,-1828,-1750,
    -1638,-1510,-1385,-1277,-1195,-1135,-1084,-1028,-954,-855,-731,-591,
    -437,-274,-104,78,269,468,661,833,972,1071,1133,1172,
    1210,1268,1357,1477,1612,1745,1853,1927,1966,1985,2001,2032,
    2087,2165,2255,2342,2411,2458,2480,2488,2488,2482,2467,2429,
    2357,2238,2075,1878,1668,1471,1306,1187,1108,1054,1006,945,
    860,751,627,502,388,283,181,70,-73,-255,-479,-729,
    -981,-1212,-1396,-1523,-1592,-1617,-1616,-1609,-1607,-1613,-1623,-1628,
    -1622,-1604,-1580,-1557,-1547,-1556,-1580,-1610,-1628,-1622,-1579,-1496,
    -1381,-1249,-1117,-1002,-915,-856,-820,-794,-766,-724,-665,-587,
    -501,-416,-347,-307,-302,-333,-394,-473,-553,-616,-646,-634,
    -577,-484,-366,-239,-122,-24,50,106,150,194,246,307,
    376,439,486,508,505,486,467,467,503,581,698,835,
    974,1093,1178,1226,1241,1237,1224,1208,1189,1154,1090,991,
    852,683,498,310,130,-40,-206,-380,-572,-786,-1020,-1261,
    -1498,-1717,-1913,-2084,-2234,-2365,-2472,-2550,-2584,-2565,-2492,-2372,
    -2227,-2082,-1961,-1877,-1826,-1795,-1761,-1698,-1593,-1446,-1273,-1097,
    -944,-832,-765,-731,-708,-672,-596,-469,-289,-61,198,471,
    742,998,1231,1434,1607,1751,1867,1958,2024,2066,2080,2064,
    2013,1930,1817,1684,1554,1444,1376,1363,1406,1498,1618,1745,
    1859,1948,2004,2025,2023,1999,1953,1881,1775,1633,1457,1259,
    1059,880,737,637,576,536,489,415,296,129,-78,-305,
    -529,-731,-901,-1033,-1130,-1202,-1256,-1298,-1334,-1373,-1420,-1478,
    -1556,-1654,-1771,-1893,-2009,-2100,-2152,-2159,-2118,-2040,-1932,-1810,
    -1684,-1559,-1437,-1318,-1201,-1089,-986,-897,-829,-787,-769,-768,
    -773,-774,-760,-725};
int data3[512] = {-2126,-2096,-2041,-2274,-2257,-2368,-2315,-2330,-2266,-2251,-2205,-2158,
    -2098,-2026,-1909,-1768,-1637,-1510,-1380,-1254,-1137,-1037,-950,-861,
    -841,-886,-937,-1101,-1279,-1443,-1671,-1810,-1829,-1837,-1758,-1595,
    -1466,-1331,-1206,-1136,-1089,-1095,-1135,-1149,-1175,-1210,-1195,-1155,
    -1097,-987,-885,-776,-609,-491,-387,-277,-202,-127,-53,3,
    47,130,219,317,448,559,671,786,827,822,824,780,
    711,647,563,513,469,424,374,299,209,96,-42,-227,
    -441,-682,-957,-1215,-1462,-1655,-1816,-1911,-1942,-1914,-1887,-1893,
    -1941,-2074,-2224,-2432,-2651,-2852,-2874,-2776,-2748,-2526,-2255,-2098,
    -1922,-1803,-1647,-1485,-1307,-1089,-880,-670,-498,-385,-326,-331,
    -338,-337,-336,-303,-249,-166,-75,11,109,226,360,516,
    684,871,1035,1153,1306,1437,1527,1629,1787,1902,2008,2108,
    2149,2152,2086,1990,1846,1715,1588,1469,1382,1304,1247,1198,
    1144,1112,1079,1027,961,864,757,644,494,395,288,162,
    64,-70,-218,-405,-646,-900,-1178,-1444,-1664,-1887,-2089,-2237,
    -2390,-2506,-2572,-2613,-2591,-2540,-2455,-2354,-2244,-2121,-2005,-1885,
    -1790,-1721,-1643,-1594,-1587,-1622,-1679,-1745,-1822,-1873,-1903,-1903,
    -1839,-1771,-1697,-1618,-1569,-1519,-1479,-1421,-1362,-1320,-1284,-1267,
    -1266,-1298,-1331,-1346,-1354,-1321,-1280,-1233,-1190,-1199,-1251,-1327,
    -1418,-1496,-1527,-1504,-1439,-1317,-1163,-1012,-855,-739,-630,-537,
    -430,-308,-204,-84,31,156,274,382,486,563,634,688,
    725,744,748,755,755,738,682,588,450,277,71,-140,
    -331,-473,-555,-601,-643,-709,-816,-992,-1221,-1495,-1764,-2019,
    -2221,-2379,-2490,-2562,-2608,-2597,-2565,-2489,-2384,-2257,-2119,-2010,
    -1935,-1914,-1936,-1959,-1962,-1914,-1805,-1636,-1436,-1228,-1027,-863,
    -721,-587,-443,-262,-55,166,389,587,740,843,918,968,
    1016,1074,1143,1221,1281,1325,1349,1339,1325,1318,1326,1363,
    1417,1478,1533,1576,1602,1595,1570,1530,1482,1432,1372,1308,
    1241,1177,1114,1053,998,937,873,806,721,624,518,396,
    262,115,-51,-231,-427,-629,-822,-1002,-1150,-1259,-1338,-1395,
    -1439,-1487,-1542,-1604,-1665,-1714,-1745,-1754,-1748,-1741,-1745,-1772,
    -1824,-1895,-1971,-2033,-2063,-2049,-1993,-1897,-1778,-1651,-1526,-1416,
    -1324,-1255,-1207,-1172,-1153,-1143,-1139,-1141,-1145,-1150,-1155,-1157,
    -1157,-1156,-1156,-1161,-1175,-1202,-1246,-1302,-1367,-1427,-1471,-1485,
    -1461,-1400,-1305,-1192,-1073,-961,-865,-786,-720,-664,-612,-560,
    -506,-447,-382,-304,-208,-90,48,196,341,467,566,634,
    682,724,779,854,946,1037,1104,1122,1078,978,840,698,
    584,520,510,538,578,599,581,518,418,299,178,62,
    -52,-181,-344,-551,-799,-1073,-1342,-1577,-1756,-1871,-1934,-1970,
    -2009,-2072,-2172,-2301,-2439,-2560,-2636,-2649,-2590,-2463,-2282,-2062,
    -1824,-1579,-1342,-1121,-917,-730,-558,-395,-235,-70,100,275,
    448,608,744,852,929,985,1031,1081,1149,1238,1345,1461,
    1571,1664,1735,1788,1832,1879,1937,2005};
int data4[512] = {200,1300,54,382,-483,-25,-1702,-616,-679,-4803,-2327,351,4567,
    9940,9679,4341,2980,-489,-1377,-5389,-6514,-7554,-4812,-5168,-6014,-5978,-4536,
    -3872,-1891,-911,-1180,3769,4205,4920,2740,7011,406,2015,191,-2626,-708,
    4546,7399,13005,14261,7076,4883,2179,1587,-3054,-3875,-8971,-8462,-8015,-6182,
    -8919,-3552,-6929,-3420,-5363,-4672,-1237,6406,7944,2412,8477,3312,4954,2044,
    -1565,-4778,-2862,-1264,9561,15585,11448,5037,2863,2599,-2787,-4556,-6972,-10842,
    -11643,-9641,-9017,-3932,-7661,-4819,-7255,-6075,-7325,3509,7292,5574,9030,5438,
    5012,1794,272,-1685,-262,-1099,7067,14560,16440,9369,5582,4008,2148,273,
    851,-5294,-5117,-9710,-9786,-6360,-6077,-5827,-7094,-4626,-9766,-256,3998,4639,
    4983,4876,4687,3987,1187,55,-368,-1715,1791,10539,15866,9966,4454,4311,
    1516,-331,831,-4862,-4362,-11358,-10762,-11432,-8628,-10512,-7913,-6712,-12090,-4778,
    895,2902,4454,5771,5616,7469,5980,5520,4029,3187,2239,10018,17649,14657,
    5844,5038,-111,-76,-987,-4118,-5612,-11379,-11615,-13283,-7735,-10964,-7257,-6390,
    -10525,-7928,-1928,156,1929,1793,4122,5348,3714,3152,5044,4550,2094,7434,
    17814,18799,10821,9702,2741,3073,-218,161,-3751,-7655,-11047,-12208,-7944,-10799,
    -6888,-3536,-7333,-7590,-3370,-414,1741,864,3417,4429,4511,3750,5792,5125,
    2222,2734,15403,19224,14929,11438,5048,4967,521,940,-3339,-4629,-10926,-12217,
    -9565,-12869,-10925,-6240,-9584,-10778,-10097,-5972,-3577,-3309,-1668,722,906,-399,
    2411,4543,1151,-2628,11271,17460,18715,14571,10354,7231,3266,2846,-607,-2324,
    -10354,-12348,-9970,-13406,-12497,-6790,-7568,-8755,-9230,-4184,-1658,-264,325,3493,
    3082,2490,2924,7792,4585,-1665,7629,15085,21035,17915,16271,11723,8948,6125,
    3205,2259,-5885,-11228,-9042,-13319,-13663,-10353,-8454,-10677,-11944,-9009,-5390,-3727,
    -2959,60,61,-601,-3085,4669,2365,-4419,-475,8525,16360,15531,15875,11562,
    9840,6505,4677,5592,-2843,-9059,-7288,-12019,-14210,-12200,-7921,-9641,-11108,-10593,
    -6297,-5417,-3777,-1327,1373,756,-3643,4610,4013,-1370,-2995,5203,13764,16921,
    18069,14152,12503,9369,6876,10257,3710,-2512,-3601,-7146,-10440,-10895,-7814,-8307,
    -10113,-11464,-7392,-7230,-4956,-5039,390,145,-4557,1734,4278,572,-5220,231,
    8439,14470,17129,15115,13971,9605,5722,10166,5183,-737,-2828,-4442,-9501,-10720,
    -8134,-6280,-9262,-10875,-8541,-7819,-5764,-7654,-732,-800,-5319,-1703,3827,2207,
    -4776,-2813,3492,11127,16314,15533,15852,11141,6389,9547,7006,1194,-2354,-3479,
    -7925,-9704,-9604,-6273,-8572,-9512,-10043,-8222,-6395,-9049,-2082,-360,-4291,-4770,
    1452,2714,-3492,-4611,-955,6659,13430,14001,16973,12635,7586,9622,9918,4798,
    823,-712,-4408,-7862,-8364,-3842,-6147,-6502,-8587,-5808,-5172,-8090,-2864,973,
    -1645,-4641,716,4157,-1372,-4400,-4670,2081,9149,11328,15994,12758,7584,7681,
    9364,5451,2092,-871,-3018,-8152,-9540,-6728,-7052,-6152,-9816,-7299,-6107,-8897,
    -6299,-1049,-1394,-5917,-2533,2543,-124,-3497,-7610,-2121,4511,8349,14727,14741,
    9544,8258,10435,8232,5454,1768,1066,-5352,-7244};
int data5[512*7] = {200,1300,54,382,-483,-25,-1702,-616,-679,-4803,-2327,351,4567,9940,
    9679,4341,2980,-489,-1377,-5389,-6514,-7554,-4812,-5168,-6014,-5978,-4536,-3872,-1891,
    -911,-1180,3769,4205,4920,2740,7011,406,2015,191,-2626,-708,4546,7399,13005,
    14261,7076,4883,2179,1587,-3054,-3875,-8971,-8462,-8015,-6182,-8919,-3552,-6929,-3420,
    -5363,-4672,-1237,6406,7944,2412,8477,3312,4954,2044,-1565,-4778,-2862,-1264,9561,
    15585,11448,5037,2863,2599,-2787,-4556,-6972,-10842,-11643,-9641,-9017,-3932,-7661,-4819,
    -7255,-6075,-7325,3509,7292,5574,9030,5438,5012,1794,272,-1685,-262,-1099,7067,
    14560,16440,9369,5582,4008,2148,273,851,-5294,-5117,-9710,-9786,-6360,-6077,-5827,
    -7094,-4626,-9766,-256,3998,4639,4983,4876,4687,3987,1187,55,-368,-1715,1791,
    10539,15866,9966,4454,4311,1516,-331,831,-4862,-4362,-11358,-10762,-11432,-8628,-10512,
    -7913,-6712,-12090,-4778,895,2902,4454,5771,5616,7469,5980,5520,4029,3187,2239,
    10018,17649,14657,5844,5038,-111,-76,-987,-4118,-5612,-11379,-11615,-13283,-7735,-10964,
    -7257,-6390,-10525,-7928,-1928,156,1929,1793,4122,5348,3714,3152,5044,4550,2094,
    7434,17814,18799,10821,9702,2741,3073,-218,161,-3751,-7655,-11047,-12208,-7944,-10799,
    -6888,-3536,-7333,-7590,-3370,-414,1741,864,3417,4429,4511,3750,5792,5125,2222,
    2734,15403,19224,14929,11438,5048,4967,521,940,-3339,-4629,-10926,-12217,-9565,-12869,
    -10925,-6240,-9584,-10778,-10097,-5972,-3577,-3309,-1668,722,906,-399,2411,4543,1151,
    -2628,11271,17460,18715,14571,10354,7231,3266,2846,-607,-2324,-10354,-12348,-9970,-13406,
    -12497,-6790,-7568,-8755,-9230,-4184,-1658,-264,325,3493,3082,2490,2924,7792,4585,
    -1665,7629,15085,21035,17915,16271,11723,8948,6125,3205,2259,-5885,-11228,-9042,-13319,
    -13663,-10353,-8454,-10677,-11944,-9009,-5390,-3727,-2959,60,61,-601,-3085,4669,2365,
    -4419,-475,8525,16360,15531,15875,11562,9840,6505,4677,5592,-2843,-9059,-7288,-12019,
    -14210,-12200,-7921,-9641,-11108,-10593,-6297,-5417,-3777,-1327,1373,756,-3643,4610,4013,
    -1370,-2995,5203,13764,16921,18069,14152,12503,9369,6876,10257,3710,-2512,-3601,-7146,
    -10440,-10895,-7814,-8307,-10113,-11464,-7392,-7230,-4956,-5039,390,145,-4557,1734,4278,
    572,-5220,231,8439,14470,17129,15115,13971,9605,5722,10166,5183,-737,-2828,-4442,
    -9501,-10720,-8134,-6280,-9262,-10875,-8541,-7819,-5764,-7654,-732,-800,-5319,-1703,3827,
    2207,-4776,-2813,3492,11127,16314,15533,15852,11141,6389,9547,7006,1194,-2354,-3479,
    -7925,-9704,-9604,-6273,-8572,-9512,-10043,-8222,-6395,-9049,-2082,-360,-4291,-4770,1452,
    2714,-3492,-4611,-955,6659,13430,14001,16973,12635,7586,9622,9918,4798,823,-712,
    -4408,-7862,-8364,-3842,-6147,-6502,-8587,-5808,-5172,-8090,-2864,973,-1645,-4641,716,
    4157,-1372,-4400,-4670,2081,9149,11328,15994,12758,7584,7681,9364,5451,2092,-871,
    -3018,-8152,-9540,-6728,-7052,-6152,-9816,-7299,-6107,-8897,-6299,-1049,-1394,-5917,-2533,
    2543,-124,-3497,-7610,-2121,4511,8349,14727,14741,9544,8258,10435,8232,5454,1768,
    1066,-5352,-7244,-6568,-5274,-4191,-7736,-4583,-2766,-5830,-5002,447,2364,-3079,-1424,
    2985,2904,13,-6142,-3727,610,4986,11929,15012,9996,7960,8909,8396,5643,1952,
    1859,-4541,-6469,-8638,-6729,-5467,-8814,-7277,-4531,-6080,-7235,-2844,1782,-3075,-2824,
    16,2896,1465,-5456,-5777,-2652,1018,7806,13941,10621,8261,7437,9190,6354,3369,
    2785,-2268,-4649,-8510,-6876,-4985,-8168,-7537,-4520,-4634,-8153,-3671,2350,-970,-1746,
    -542,3776,4055,-2334,-5022,-3762,-2371,4366,12089,11327,8641,6251,9120,6336,4694,
    3724,117,-2841,-7967,-7254,-5058,-7539,-8212,-5486,-3732,-8739,-5943,-195,54,-540,
    -1093,3792,6721,1877,-2498,-2455,-3484,1726,9363,12923,10294,6686,9171,6627,5753,
    4131,1853,-632,-6605,-7398,-5538,-6572,-8653,-6079,-2652,-8222,-7307,-2492,-369,-1386,
    -2556,1628,6535,3696,-406,-1526,-4771,-1644,5196,12241,10139,7005,8713,6557,5760,
    3791,2138,279,-6028,-7937,-7030,-6623,-10164,-8030,-3366,-7485,-8202,-4786,-1032,-863,
    -2975,-1211,5382,4876,2011,365,-3323,-2809,1705,11122,11189,8679,8822,7464,7349,
    4847,3644,2407,-3243,-6870,-6847,-4780,-8964,-8172,-3312,-5379,-7745,-6029,-1718,-65,
    -2628,-2557,3949,4268,3106,1526,-1528,-3775,-1591,8374,10768,9652,9099,7976,7602,
    4944,4519,3546,-988,-6503,-7886,-5370,-9781,-10167,-5415,-5169,-8109,-8190,-4417,-602,
    -2526,-3936,1619,3755,4504,2430,460,-3518,-4045,5006,9766,11038,9715,8913,9083,
    6432,5431,4874,2653,-4194,-7025,-4573,-8521,-10584,-6983,-4716,-6656,-8304,-6340,-894,
    -2117,-4200,-223,3064,4903,3455,3341,-1961,-5565,1129,7214,11014,10148,9985,9900,
    7192,6103,5626,4978,-2274,-6246,-4518,-7406,-11005,-9106,-6112,-6402,-8732,-8611,-2145,
    -2176,-3966,-2198,1630,4503,3668,5284,451,-4851,-2258,3586,9191,9516,9742,9969,
    8424,6772,5664,7234,909,-4710,-4572,-5969,-10542,-10333,-8036,-6206,-9012,-10859,-4734,
    -2907,-3896,-4238,-69,2973,2859,6054,2936,-3205,-4291,472,6802,8480,9598,9737,
    9307,7102,6194,8761,3929,-2535,-3260,-4073,-8891,-10114,-9055,-5365,-7706,-10697,-6367,
    -3409,-3055,-4833,-768,2103,2659,6427,5562,20,-3832,-1485,4545,7887,9945,9900,
    10750,7928,6485,9344,7555,702,-1767,-2551,-6093,-9225,-10141,-5662,-6918,-10112,-8410,
    -4549,-3015,-5790,-2888,-60,784,4626,5875,2328,-3575,-3818,796,5126,8023,8706,
    10688,8051,5702,8382,8948,2577,-249,-1582,-3934,-8239,-10520,-6474,-6519,-9019,-9632,
    -5638,-3068,-5483,-3509,-1242,-567,3134,5989,5054,-1234,-3725,-1769,3003,6546,8082,
    10892,9592,6108,8050,10191,5351,1721,-879,-1514,-5601,-9623,-7324,-5755,-7281,-10050,
    -6900,-3364,-5297,-4238,-2335,-1484,870,4648,6263,1069,-2949,-3572,758,4376,6640,
    10049,10633,6344,7337,9969,7325,3405,-323,-237,-4165,-8667,-8040,-5963,-6481,-9859,
    -8006,-4399,-5514,-4829,-3609,-2477,-1398,3116,6367,3396,-1393,-4196,-1399,2157,4745,
    8862,11363,7263,7265,9493,9391,6029,1455,1289,-1714,-6841,-8650,-6609,-5276,-9168,
    -8689,-5230,-5206,-5175,-4281,-2901,-3123,677,4932,5145,665,-4041,-3588,-307,1688,
    6346,10590,7657,6650,8032,10069,7750,3212,2526,524,-4350,-8099,-7020,-4998,-8574,
    -8628,-6061,-5052,-5188,-4215,-2899,-3689,-1120,3405,5865,3151,-2031,-3839,-1443,-713,
    3927,9314,8565,6987,7117,10193,9073,4925,3629,2463,-1276,-6583,-7018,-4732,-7525,
    -8880,-7522,-5180,-5382,-4396,-3112,-3592,-2754,1165,5112,5085,-17,-3398,-1947,-2602,
    1171,6809,8611,6929,5973,9029,9727,6445,4374,3391,1361,-4813,-6545,-4810,-6361,
    -8634,-8094,-5898,-6190,-5006,-3375,-3475,-3780,-866,3499,6151,2086,-1849,-1858,-3596,
    -1189,4285,8142,7327,5416,7963,9834,7939,5497,4532,3765,-2378,-5644,-4945,-5167,
    -8000,-8456,-6608,-6375,-5901,-4267,-3710,-4374,-2925,1183,6259,3759,163,-1046,-3228,
    -3123,1160,6668,7762,5255,6793,9060,8910,6129,5147,5420,348,-4255,-4814,-4371,
    -7173,-8341,-6993,-6473,-6656,-4649,-3841,-4329,-4569,-1324,5067,4942,2403,588,-1976,
    -3721,-1145,4772,7499,5554,6010,8035,9508,6680,5465,6036,2702,-2703,-4472,-3948,
    -6284,-8194,-7501,-6736,-7262,-5639,-4562,-3891,-5553,-4063,2248,4712,3326,1795,-594,
    -3321,-3390,2304,6693,6145,5356,7090,9876,7753,6202,6755,5170,-360,-3437,-3208,
    -4727,-7376,-7216,-6640,-7260,-6259,-4924,-3336,-5377,-5459,-151,3768,3947,3048,1245,
    -1765,-4172,141,5015,6316,5133,6200,9149,8384,6559,6831,6599,1920,-2347,-3138,
    -4071,-7161,-7431,-7270,-7432,-7471,-6207,-3838,-5124,-6832,-2971,1624,3600,3113,2425,
    -491,-4431,-2418,2668,5993,4906,5141,8262,9089,7035,6994,7796,4595,-280,-1999,
    -2795,-5795,-6807,-7064,-6727,-7383,-6739,-3751,-4072,-6636,-4560,-126,3253,3442,4116,
    1687,-2891,-3687,559,5104,5238,4959,7493,9399,7853,7373,8404,6834,1766,-686,
    -1821,-4408,-6443,-7161,-6909,-7617,-7864,-4636,-3914,-6203,-6311,-2542,1572,2808,4367,
    3254,-964,-4294,-2075,2933,4574,4042,5966,8802,8127,6922,8190,8342,3708,602,
    -936,-3108,-5775,-7061,-6774,-7511,-8951,-6057,-4227,-5402,-7276,-4594,-652,1700,3927,
    4490,1323,-3289,-3480,958,3873,3520,4808,7772,8509,7085,8372,9207,6024,2541,
    573,-1504,-4137,-6404,-6208,-6870,-8592,-7065,-4708,-4735,-7159,-6009,-2695,112,2683,
    4667,3219,-1517,-4061,-1052,2663,2990,3575,6250,8137,6732,7585,9254,7739,4168,
    1907,44,-2577,-5706,-5705,-6488,-8268,-8265,-5603,-4483,-6728,-7008,-4573,-1815,954,
    4041,4573,568,-3814,-2610,1034,2395,2465,4886,7341,6646,6738,8686,8388,5843,
    3280,1438,-1043,-4370,-5092,-5702,-7306,-8546,-6418,-4404,-5684,-6786,-5600,-3359,-820,
    2779,5121,2837,-2192,-3243,-420,1899,1855,3776,6610,6959,6464,8274,8970,7349,
    4481,3047,824,-2551,-4278,-4709,-6150,-8243,-7193,-4828,-5059,-6571,-6274,-4611,-2691,
    606,4526,4239,-180,-3291,-1677,835,1116,2230,5344,6479,6144,7394,8797,8138,
    5537,4107,1946,-1278,-3493,-4038,-5214,-7650,-7734,-5654,-4785,-5995,-6417,-5655,-4257,
    -1570,3056,4810,1904,-2464,-2465,-181,694,1100,4067,5851,6021,6565,8510,8759,
    6856,5364,3593,573,-2495,-3382,-4039,-6496,-7846,-6264,-4660,-5285,-6175,-5844,-5214,
    -3445,859,4751,3757,-730,-2665,-1051,3,186,2590,4973,5560,5724,7783,8791,
    7620,6281,4857,2266,-1179,-2632,-3473,-5564,-7669,-7048,-5377,-5050,-5955,-5795,-5887,
    -5008,-1707,3363,4444,1293,-2107,-1576,-691,-463,1048,3864,4999,5256,6963,8554,
    8171,7078,6148,4154,636,-1642,-2345,-3991,-6719,-7505,-5863,-4979,-5506,-5572,-5706,
    -5908,-3896,1354,4577,2980,-891,-1541,-900,-876,-184,2558,4223,4479,5799,7782,
    8242,7276,6821,5385,2253,-747,-1583,-3020,-5578,-7386,-6362,-5655,-5414,-5637,-5417,
    -6357,-5478,-1392,3365,3803,820,-1252,-1070,-1085,-862,1352,3323,3815,4794,6816,
    7977,7490,7286,6586,4002,784,-668,-1668,-4020,-6570,-6491,-5812,-5349,-5485,-4906,
    -5989,-6260,-3620,1805,4208,2488,-223,-510,-894,-1394,287,2466,3246,3882,5822,
    7315,7291,7107,7227,5269,2237,-183,-832,-3059,-5612,-6755,-6205,-5999,-5618,-4998,
    -5690,-6978,-5692,-911,3336,3391,935,-73,-589,-1374,-439,1600,2787,3356,5031,
    6881,7249,7251,7655,6554,3952,1020,177,-1743,-4142,-6287,-6157,-6256,-5745,-5017,
    -4904,-6644,-6877,-3570,1671,3609,2038,860,183,-1064,-999,904,2308,2787,4013,
    6261,6890,7291,7473,7509,5136,2276,713,-652,-3120,-5595,-6243,-6634,-6555,-5520,
    -4826,-6274,-7531,-5998,-889,2642,2445,1360,618,-662,-1159,-13,1536,2146,3181,
    5209,6233,6960,7132,7983,6278,3781,1497,366,-1951,-4378,-5813,-6327,-6910,-5926,
    -4821,-5397,-7087,-7378,-3266,1317,2621,1967,1399,72,-897,-592,1235,1598,2638,
    4184,5880,6472,6843,7904,7260,5061,2594,1266,-734,-3293,-5096,-5919,-7097,-6361,
    -5119,-4748,-6217,-7839,-5414,-613,1973,2335,1983,861,-428,-764,736,1080,2262,
    3328,5226,5870,6554,7457,7801,6064,3785,2148,559,-2075,-3944,-5290,-6853,-6685,
    -5794,-4663,-5462,-7649,-7249,-2900,466,2137,2067,1707,-138,-694,23,595,1297,
    2412,4335,5417,6019,6974,7932,7109,4998,3156,1788,-872,-2708,-4274,-6143,-6818,
    -6411,-4931,-4700,-6912,-7966,-5025,-1354,1395,2023,2406,458,-187,-198,517,762,
    1815,3408,4913,5629,6740,7745,7964,6266,4387,3029,453,-1369,-3260,-5259,-6582,
    -6771,-5683,-4373,-6232,-7912,-6914,-3254,-254,1553,2500,1193,116,-300,211,386,
    959,2379,3948,4847,5803,6989,8101,6964,5378,4136,1592,-158,-2075,-4187,-5954,
    -6957,-6403,-4427,-5503,-7172,-8024,-4912,-2081,823,2314,2004,688,119,183,429,
    487,1772,3173,4422,5293,6333,7908,7465,6532,5280,2945,1084,-684,-2977,-4717,
    -6694,-6716,-5125,-4828,-6335,-8182,-6424,-3968,-718,1470,2046,1103,262,20,176,
    16,930,2120,3604,4483,5379,7217,7219,6927,6027,4052,2092,644,-1753,-3196,
    -5942,-6564,-5848,-4365,-5604,-7394,-7252,-5480,-2543,456,1800,1642,709,388,265,
    44,565,1384,3029,3920,4840,6649,7076,7226,6976,4990,3232,1657,-432,-2001,
    -4667,-6165,-6484,-4604,-5062,-6571,-7564,-6778,-4367,-1056,976,1637,806,468,166,
    -13,245,512,2217,3067,4113,5850,6629,6979,7571,5789,4422,2296,789,-981,
    -3094,-5526,-6493,-5061,-4698,-5798,-6984,-7340,-5797,-2716,120,1494,1134,987,429,
    275,299,267,1756,2596,3476,5380,6153,6778,7824,6674,5505,3260,2018,-9,
    -1724,-4488,-6218,-5706,-4768,-5296,-6310,-7591,-6997,-4606,-1450,628,944,1175,400,
    389,146,-78,911,1869,2380,4645,5178,6206,7275,7236,6207,4269,2808,1063,
    -419,-3064,-5564,-5835,-5006,-4950,-5470,-7033,-7431,-6162,-2885,-313,689,1236,738,
    695,437,-100,669,1374,1620,3925,4531,5663,6519,7469,6665,5247,3608,2105,
    704,-1437,-4320,-5411,-5256,-4896,-4813,-6137,-7141,-7181,-4291,-1795,-23,950,808,
    592,661,-393,483,651,947,2816,3844,4811,5695,7183,6984,5928,4352,2832,
    1586,-119,-3075,-4743,-5362,-4992,-4728,-5269,-6619,-7693,-5597,-3100,-866,614,866,
    840,1056,-331,688,401,632,1840,3342,4067,4865,6489,7077,6515,5225,3584,
    2537,1078,-1658,-3580,-5026,-5008,-4924,-4591,-5925,-7510,-6767,-4419,-2232,79,500,
    999,1090,-41,717,311,203,956,2643,3442,4033,5626,6761,6738,5901,4198,
    3320,2076,-314,-2403,-4351,-4897,-5174,-4368,-5141,-6957,-7280,-5632,-3715,-777,-245,
    1200,1201,379,711,750,195,451,2009,3036,3441,4818,6247,6839,6576,4796,
    4107,2923,1108,-1200,-3128,-4551,-5279,-4494,-4348,-6356,-7102,-6634,-4957,-1960,-1082,
    803,1187,590,691,1056,331,26,1318,2535,2849,4000,5416,6584,6957,5475,
    4758,3618,2278,-92,-1849,-3743,-5118,-4890,-3863,-5769,-6421,-7230,-5880,-3313,-1911,
    105,1136,711,647,1262,823,-114,749,2142,2470,3437,4463,6148,6881,6109,
    5198,4336,3159,998,-778,-2592,-4873,-5074,-3861,-5171,-5745,-7327,-6567,-4561,-2917,
    -904,803,800,502,1172,1196,-97,255,1659,2045,2938,3565,5522,6538,6572,
    5556,5008,3953,2174,174,-1056,-4259,-4866,-4184,-4615,-5122,-6875,-6994,-5591,-4073,
    -2059,136,788,403,927,1602,194,-42,919,1700,2405,2791,4481,5892,6455,
    5810,5326,4671,3012,1041,423,-3073,-4311,-4453,-4230,-4518,-6142,-6903,-6159,-5057,
    -3238,-824,531,258,691,1873,678,133,420,1565,2051,2494,3634,5454,6281,
    6332,5596,5587,3811,2065,1525,-1633,-3527,-4439,-4221,-4150,-5501,-6655,-6451,-5849,
    -4459,-2099,2,-107,275,1606,1061,260,-56,1070,1621,1931,2586,4523,5690,
    6280,5527,6270,4371,2859,2292,-128,-2509,-4172,-4331,-3936,-4950,-6268,-6529,-6423,
    -5613,-3519,-705,-535,-91,1064,1454,593,-80,591,1491,1542,2160,3514,5211,
    5997,5631,6817,5164,3735,3121,1434,-950,-3210,-4078,-3629,-4301,-5510,-6174,-6372,
    -6352,-4801,-1641,-970,-492,515,1575,999,53,109,1294,1181,1770,2325,4578,
    5323,5284,6754,5732,4297,3613,2493,425,-2189,-3832,-3531,-3945,-5028,-5989,-6021,
    -6875,-6025,-3104,-1546,-1186,-221,1108,1365,152,-138,885,997,1480,1430,3931,
    4688,4876,6550,6300,4975,4161,3334,1937,-692,-2926,-3235,-3442,-4378,-5484,-5363,
    -6744,-6757,-4341,-2018,-1652,-802,493,1696,504,83,389,1016,1245,769,3088,
    4045,4224,5879,6477,5438,4435,3622,3000,557,-1933,-3044,-2995,-3997,-5144,-4971,
    -6148,-7196,-5538,-3009,-2139,-1587,-322,1434,684,239,-127,927,1106,251,2181,
    3475,3586,5091,6356,5971,4886,4022,3814,1858,-630,-2557,-2462,-3464,-4662,-4619,
    -5235,-7135,-6299,-4130,-2393,-2235,-971,1013,1008,690,-272,916,1199,108,1515,
    3116,3146,4322,5919,6537,5403,4482,4267,3190,727,-1705,-2013,-2911,-4326,-4358,
    -4460,-6518,-6867,-5285,-2821,-2854,-1773,143,1044,1046,-361,678,1247,-21,758,
    2527,2692,3435,4993,6556,5686,4828,4369,4168,1953,-656,-1598,-2187,-4049,-4156,
    -4049,-5570,-6975,-6177,-3533,-3373,-2468,-876,753,1225,-183,419,1400,123,311,
    1846,2501,2734,4028,6070,6025,5186,4508,4774,3201,547,-843,-1270,-3410,-3948,
    -3853,-4446,-6687,-6805,-4384,-3717,-3008,-1933,168,1293,30,142,1453,322,18,
    1091,2306,2151,3056,5045,6178,5455,4664,4893,4293,1579,-4,-570,-2638,-3724,
    -3814,-3534,-6024,-7082,-5340,-4176,-3445,-2931,-832,1144,269,-33,1298,695,-70,
    475,2061,1930,2329,3953,5936,5702,4878,4910,5224,2644,990,188,-1499,-3249,
    -3790,-2854,-4994,-6715,-6039,-4603,-3687,-3662,-1954,864,408,-15,1048,1261,25,
    104,1579,1930,1802,2948,5169,5872,4989,4839,5662,3569,1876,845,-329,-2660,
    -3751,-2677,-4074,-6093,-6559,-5365,-3991,-4277,-3304,-99,180,-177,560,1489,205,
    -179,945,1885,1403,2058,3978,5782,5042,4753,5686,4434,2703,1401,864,-1708,
    -3469,-2645,-3219,-5193,-6495,-5966,-4113,-4545,-4253,-1258,74,-347,159,1488,727,
    -250,497,1759,1423,1458,2762,5291,4946,4662,5473,5082,3442,1871,1765,-491,
    -2833,-2629,-2594,-4058,-5920,-6301,-4314,-4569,-4765,-2519,-321,-626,-213,1130,1141,
    -180,123,1253,1532,1040,1651,4425,4735,4584,5090,5510,4096,2324,2433,786,
    -2049,-2539,-2397,-3060,-5235,-6444,-4808,-4510,-5187,-3780,-1035,-870,-679,649,1481,
    86,-96,793,1783,953,988,3455,4462,4531,4796,5779,4804,2868,2946,2025,
    -915,-2167,-2332,-2206,-4357,-6188,-5280,-4400,-5390,-4705,-2108,-1035,-1227,101,1487,
    558,-150,207,1787,1023,580,2348,3883,4248,4324,5577,5324,3320,3226,2839,
    450,-1559,-2301,-1661,-3315,-5671,-5587,-4355,-5283,-5361,-3228,-1375,-1744,-582,1054,
    1067,-18,-233,1566,1191,446,1392,3240,3946,3941,5330,5806,3810,3498,3399,
    1914,-646,-2026,-1389,-2141,-4940,-5619,-4382,-4933,-5739,-4234,-1882,-2094,-1250,402,
    1402,276,-444,1181,1461,590,780,2485,3597,3471,4880,6002,4414,3701,3629,
    3096,511,-1602,-1370,-1241,-4031,-5455,-4599,-4535,-5936,-5116,-2756,-2459,-1961,-621,
    1284,612,-540,512,1467,828,355,1643,3218,2842,4212,5798,4992,3836,3731,
    3831,1937,-860,-1285,-558,-2903,-5038,-4752,-4107,-5766,-5645,-3644,-2697,-2539,-1595,
    849,961,-365,1,1307,1198,241,1010,2826,2404,3500,5375,5498,4125,3750,
    4188,3298,170,-1026,-256,-1618,-4349,-4849,-3783,-5339,-5881,-4560,-3023,-2939,-2541,
    -25,1133,-58,-295,889,1512,222,591,2329,2130,2654,4695,5659,4596,3739,
    4164,4341,1468,-572,-111,-398,-3396,-4717,-3653,-4725,-5825,-5284,-3434,-3162,-3284,
    -1057,947,307,-437,331,1717,433,307,1847,2034,1869,3762,5378,5073,3772,
    3929,4848,2798,151,-67,545,-2200,-4260,-3620,-3980,-5441,-5691,-3920,-3256,-3803,
    -2164,239,638,-415,-208,1505,712,120,1373,1898,1380,2745,4772,5471,3984,
    3687,4926,4049,1086,-22,1088,-963,-3569,-3658,-3399,-4862,-5868,-4477,-3334,-4118,
    -3163,-813,790,-277,-562,1128,1062,17,988,1780,1270,1789,3826,5562,4373,
    3543,4617,4968,2225,281,1328,180,-2577,-3601,-3113,-4083,-5777,-5003,-3558,-4118,
    -3988,-2053,469,-51,-832,566,1220,58,638,1564,1365,1024,2666,5159,4782,
    3553,4067,5436,3331,853,1353,1139,-1387,-3278,-3065,-3317,-5450,-5349,-3938,-3863,
    -4531,-3308,-352,98,-969,30,1135,219,345,1252,1512,632,1567,4294,5069,
    3767,3518,5418,4381,1624,1334,1784,-78,-2634,-2961,-2651,-4802,-5515,-4403,-3646,
    -4660,-4351,-1492,63,-998,-424,868,432,121,895,1662,674,715,3029,5024,
    4074,3147,5001,5165,2515,1451,2038,1192,-1788,-2744,-2290,-3851,-5283,-4738,-3544,
    -4369,-4972,-2715,-374,-993,-719,436,563,31,509,1625,927,214,1634,4462,
    4363,2912,4273,5406,3439,1746,2072,2138,-634,-2210,-1980,-2950,-4895,-5021,-3687,
    -3879,-5208,-3845,-1093,-982,-939,-24,608,7,194,1391,1334,175,530,3499,
    4537,2943,3607,5205,4272,2173,2046,2714,585,-1431,-1702,-2109,-4008,-4925,-3862,
    -3262,-4946,-4638,-2061,-1148,-1172,-480,510,44,-118,942,1609,450,-272,2215,
    4215,3029,3007,4667,4826,2615,1991,2835,1612,-595,-1325,-1506,-3005,-4631,-4130,
    -2887,-4388,-4987,-3060,-1536,-1359,-863,254,129,-300,461,1634,1031,-530,1025,
    3578,3253,2629,3907,5045,3330,2068,2777,2426,323,-791,-1018,-2003,-4055,-4263,
    -2736,-3669,-4863,-3833,-2026,-1622,-1229,-85,197,-319,-38,1368,1568,-358,94,
    2574,3291,2374,3126,4795,3853,2248,2652,2878,1170,-251,-661,-1102,-3285,-4214,
    -2835,-3051,-4472,-4353,-2681,-1885,-1587,-507,125,-212,-497,851,1860,136,-468,
    1512,3057,2276,2420,4258,4221,2489,2447,3039,1987,391,-300,-311,-2340,-3914,
    -3062,-2534,-3805,-4398,-3198,-2196,-1888,-944,-20,39,-772,273,1901,854,-588,
    564,2601,2306,1953,3581,4326,2863,2374,2928,2593,1046,70,229,-1289,-3371,
    -3253,-2311,-3120,-4228,-3606,-2601,-2154,-1441,-402,220,-828,-353,1523,1423,-345,
    -141,1900,2272,1583,2810,4146,3191,2371,2641,2931,1635,375,630,-293,-2591,
    -3284,-2333,-2485,-3821,-3756,-2990,-2383,-1934,-837,233,-663,-906,904,1745,175,
    -541,1122,2107,1381,2133,3639,3372,2463,2301,3020,2171,704,808,534,-1618,
    -3073,-2408,-2066,-3201,-3590,-3227,-2556,-2356,-1361,17,-381,-1193,90,1673,794,
    -577,403,1812,1258,1492,2981,3452,2611,2064,2885,2641,1088,890,1123,-504,
    -2578,-2429,-1870,-2575,-3327,-3309,-2682,-2631,-1964,-367,-132,-1228,-685,1252,1282};

int main() {
    cout<<endl<<"PSOLA Testing: "<<endl<<endl;
    long fs = 8000;
    float pitch = 523.251;
    float desiredPitch = 587.330;
    PSOLA psola(windowLen);
    //psola.pitchCorrect(data4,fs,pitch,desiredPitch);
    for (int i = 0; i < 7; i++) {
        psola.pitchCorrect(data5+i*windowLen,fs,pitch,desiredPitch);
    }
    for (int i = 0; i < windowLen*7; i++) {
        if (i % 15 == 0) {
            cout <<"..." << endl;
        }
        cout << data5[i] << ",";
        
    }
    cout << endl << endl;
    
    
    return 0;
}
