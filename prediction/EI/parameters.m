ai = 0;
Li = 0;
vi = 0;
load('TS041_03oct2010_05_34_02_Seizure.mat');
CH26_sz = record_RMpt2(26,:);
for a=.5:.05:1
    ai = ai+1;
    for L=.05:.05:.45
        Li = Li+1;
        for v = .05:.05:.45
            vi = vi+1;
            %for t = 1:4
                %TEST 1 seizure from 191 to 405
                predict26 = epin(CH26_sz,v,L);
                WTP = sum(predict26(151:191));
                WFP = sum(predict26(1:150)) + sum(predict26(416:end));
                val(ai,Li,vi) = a*WTP - (1-a)*WFP;
            %end
        end
    end
end
