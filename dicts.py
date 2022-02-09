hydropathy = {'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5,'X':0.0,'B':0.0}
volume = {'I':166.7,'V':140.,'L':166.7,'F':189.9,'C':108.5,'M':162.9,'A':88.6,'G':60.1,'T':116.1,'S':89.0,'W':227.8,'Y':193.6,'P':112.7,'H':153.2,'E':138.4,'Q':143.8,'D':111.1,'N':114.1,'K':168.6,'R':173.4,'X':150.0,'B':150.0}

aa_map_aliphatic = set(['I','L','V'])
aa_map_hydrophobic = set(['I','L','V','M','F','Y','W','H','K','T','C','A','G'])
aa_map_aromatic = set(['F','Y','W','H'])
aa_map_positive = set(['H','K','R'])
aa_map_polar = set(['Y','W','H','K','T','C','D','E','S','N','Q'])
aa_map_negative = set(['D','E'])
aa_map_charged = set(['H','K','R','D','E'])
aa_map_small = set(['P','G','C','A','S','N','D','T','V'])
aa_map_tiny = set(['G','C','A','S'])
