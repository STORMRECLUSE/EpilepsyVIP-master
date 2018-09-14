#include "get_alarm_rate.h"

int read_model_params(const char* params_filename, double* outlier_threshold, double*weights, int CHANSIZE){
	int adaptive_rate;
	int n_chans;
	char dead_str[400];
	FILE* fid= std::fopen(params_filename,"r");
	if(!fid) {
		std::perror("Could not read patient parameters.");
		std::exit(-1);
	}
	int res = std::fscanf(fid,"channel_no: %d \nadapt_rate: %d\n",&n_chans,&adaptive_rate);
	std::fgets(dead_str,30,fid);
	if(res!=2){
		std::perror("Error in reading patient parameters: num_channels");
		std::exit(-1);
	}
	int chan =0;
	while(res>0 & chan < CHANSIZE){
		res = std::fscanf(fid,"%*d: %lf: %lf\n",outlier_threshold+chan,weights+chan);
		chan++;
	}
	if(chan <n_chans){
		//std::perror("Inconsistent file dimensions with actual threshold!");
		//std::exit(-1);
	}
	std::fclose(fid);
	return adaptive_rate;
}

//Not an actual function. (Eventually) copied into the main body of epilserv.
void initialize_pipeline(int FiltOrder, double* freq_bands, int WinStep, const char* model_filename,
						 const char* params_filename, int CHANSIZE, int WINDOWSIZE, sliding_array<double>& raw_data) {

// filter stuff
	sliding_array <double> filtered_data(CHANSIZE,WINDOWSIZE);

	// Lcutoff= 0.01 and Ucutoff = 0.1
	int FilterOrder = FiltOrder;
	double Lcutoff = freq_bands[0];
	double Ucutoff = freq_bands[1];

	double *NumC = ComputeNumCoeffs(FilterOrder);
	double *DenC = ComputeDenCoeffs(FilterOrder, Lcutoff, Ucutoff );
	double scalf = sf_bwbp(FilterOrder, Lcutoff, Ucutoff );
	int FiltOrd = 5;

	for(int k = 0; k<2*FiltOrd+1; k++)
	{
		NumC[k] = scalf*NumC[k];
	}

	// DON'T FORGET TO PARALLELIZE THIS LOOP:
	// initialize filter
	// so only the last windowstep values should be computed.
	for(int chan = 0;chan < CHANSIZE;chan ++){
		filter_vect(FilterOrder,DenC, NumC,0, WINDOWSIZE-WinStep-1,
					raw_data[chan], filtered_data[chan]);
	}



	//energy_features
	//int length;

	double **en_stat = new double*[CHANSIZE];
	en_stat[0] = new double[3*CHANSIZE];
	for(int chan = 1; chan<CHANSIZE;chan++){
		en_stat[chan] = en_stat[0] + 3*chan;//3 energy statistics per channel
	}
	//n = n_input;
	//int numWin = 1 + (n-WINDOWSIZE)/windowStep;


	//svm


	struct svm_model *seizmodel = svm_load_model(model_filename);



	//Postprocess

///Smooth Decisions / Major Votes



	double* outlier_threshold = new double[CHANSIZE];  // read threshold from file
	double* weights = new double[CHANSIZE];
	int adaptive_rate = read_model_params(params_filename,outlier_threshold,weights, CHANSIZE);   //read adaptive_rate from file

	int *alarm_sequence = new int[CHANSIZE];
	sliding_array <double> outlier_sequence(CHANSIZE,adaptive_rate);
	sliding_array <int> novelty_sequence(CHANSIZE,adaptive_rate);

}

void destroy_pipeline(double *NumC, double* DenC, double **en_stat, struct svm_model **seizmodels,
					 sliding_array <int> &novelty_sequence, sliding_array <double>  &outlier_sequence,
					 int *alarm_sequence, double *weights, double* outlier_threshold,
						sliding_array<double> &filtered_data, sliding_array<double> &raw_data, int CHANSIZE){
	free(NumC);
	free(DenC);


	delete [] en_stat[0];
	delete[] en_stat;

	for(int i = 0; i <CHANSIZE; i++){
		svm_free_and_destroy_model(seizmodels + i);

	}
	delete [] seizmodels;

	novelty_sequence.free_inside_data();
	outlier_sequence.free_inside_data();
	filtered_data.free_inside_data();
	raw_data.free_inside_data();
	delete [] alarm_sequence;
	delete [] outlier_threshold;
	free(weights);

}

void update_windows(int WindowStep,sliding_array <int> &novelty_sequence,
					sliding_array<double> &outlier_sequence,sliding_array<double >&filtered_data,
					sliding_array<double> &raw_data){
	novelty_sequence.add_access_offset(1);
	outlier_sequence.add_access_offset(1);
	filtered_data.add_access_offset(WindowStep);
	raw_data.add_access_offset(WindowStep);
}

double get_window_decision(int WINDOWSIZE, int CHANSIZE, int FilterOrder, double*NumC, double*DenC,
				sliding_array<double> &filtered_data,sliding_array<double> &raw_data,
				double **en_stat, struct svm_model **seizmodels,sliding_array<int>&novelty_sequence,
						sliding_array<double>&outlier_sequence, int adaptive_rate,
						double* outlier_thresholds,double* maj_weights, int* alarm_sequence, int num_filter) {

	//int i, j;
	int final_decision;
	double weighted_decision = 0;

	// Parallelize the alarm rate across all channels

#pragma omp parallel for
	for (int chan = 0; chan < CHANSIZE; chan++) {
		alarm_sequence[chan] = get_alarm_rate(WINDOWSIZE, FilterOrder, NumC, DenC, filtered_data[chan],
											  raw_data[chan], en_stat[chan], seizmodels[chan],
					                          novelty_sequence[chan],outlier_sequence[chan], adaptive_rate,
											  outlier_thresholds[chan],num_filter);
	//printf("Current thread number: %d\t out of %d\n", omp_get_thread_num(), omp_get_num_threads());
	}

	// Run majority_votes on alarm rate vector, return result as int

	final_decision = major_votes(maj_weights, alarm_sequence,CHANSIZE);

	// Add proper rotating storage offsets for filtered data
	for(int chan = 0; chan< CHANSIZE; chan++){
		weighted_decision += maj_weights[chan]*alarm_sequence[chan];
	}

	return weighted_decision;

}

int get_alarm_rate(int WINDOWSIZE, int FilterOrder, double *NumC, double*DenC,
		   sliding_vector <double> &filtered_data, sliding_vector <double> &raw_data,
		   double *en_stat,
           //int windowStep, int numWin, int start_index,
		   struct svm_model *seizmodel, sliding_vector <int> &novelty_sequence,
		   sliding_vector <double> &outlier_sequence, int adaptive_rate, 
		   double outlier_threshold, int WindowStep){


	//int i,j;
	int alarm_rate;
  
	/*filter*/
	filter_vect(FilterOrder, DenC, NumC,WINDOWSIZE-WindowStep, WINDOWSIZE-1, raw_data, filtered_data);
	//printf("filtered data\n");
	/*energy_features*/
	//void Energy_Stats(filtered_data, en_stat, int start_index, WINDOWSIZE, int length);
	//int WINDOW_INDEX = 0;
	//for(start_index = 0; start_index < n-WINDOWSIZE; start_index+=windowStep){
	//{
	  //for (int i = 0; i < CHANSIZE; i++){
	Energy_Stats_vect(filtered_data, en_stat, 0, WINDOWSIZE);
	//printf("Energy Statistics\n");
	  //}
	//}
	//WINDOW_INDEX++;
	//}

	/*SVM*/
	//for (i=0;i < CHANSIZE; i++){
	//for (j=0; j< numWin; j++){
	novelty_create(seizmodel, en_stat, novelty_sequence);
	//printf("Done SVM\n");
	//}
	//}

	/*Postprocess*/
	//Smooth Decisions
	alarm_rate = post_processing(novelty_sequence, outlier_sequence, adaptive_rate, outlier_threshold);
	//printf("Got Alarm Rate\n");
	return alarm_rate;
}
