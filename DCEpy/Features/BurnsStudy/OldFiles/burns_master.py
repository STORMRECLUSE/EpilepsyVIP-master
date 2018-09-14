import DCEpy.Features.BurnsStudy.test1

#
# Call using the following code in the terminal at the EpilepsyVIP directory
# python -m DCEpy.Features.BurnsStudy.burns_master.py

# Chris
inter_file = '/home/chris/Documents/Rice/senior/EpilepsyVIP/data/RMPt1/CA00100D_1-1+.edf'
ictal_file = '/home/chris/Documents/Rice/senior/EpilepsyVIP/data/RMPt1/DA00101L_1-1+.edf'

# # Emily
# inter_file = 'C:\Users\User\Documents\EpilepsySeniorDesign\Burns\CA00100D_1-1+.edf'
# ictal_file = 'C:\Users\User\Documents\EpilepsySeniorDesign\Burns\DA00101L_1-1+.edf'

#ictal_file = 'C:\Users\User\Documents\GitHub\EpilepsyVIP\DCEpy\Features\BurnsStudy\CA1353FN_small.edf'																																																																																																																																																																																																																																																																																																																				
#inter_file = 'C:\Users\User\Documents\GitHub\EpilepsyVIP\DCEpy\Features\BurnsStudy\CA1353FN_small.edf'

ictal = [277000, 345000]
inter = [0, 68000]

all_files = [ictal_file, inter_file]
centers, labels = test1.burns(all_files, ictal, inter)

count = 0
for center in centers:
	print ('Cluster '+str(count)+': ' + str(center))
	count = count