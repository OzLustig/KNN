package HomeWork4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import javax.swing.event.HyperlinkEvent;

import HomeWork4.Knn.EditMode;
import weka.core.Instances;

public class MainHW4 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception 
	{    
		Instances cancerData = loadData("cancer.txt");
		Instances glassData = loadData("glass.txt");



		// Choose Hyper Parameters by using cross validation
		int numOfFolds = 10;
		int minK = 1;
		int maxK = 20;
		double minHyperAvgErr = 1;
		double[] hyperParmArrCancer = new double[3];
		double[] hyperParmArrGlass = new double[3];
		String majorityMethod;

		// First part of the Homework.

		Knn noneKnnGlass = new Knn();
		noneKnnGlass.setEditMode(EditMode.None);

		// Performing cross validation.
		for (int K = minK; K <= maxK; K++) {
			for (int lP = 0; lP <= 3; lP++) {
				for (int majMethod = 0; majMethod <= 1; majMethod++) {
					noneKnnGlass.setK(K);
					noneKnnGlass.setLPDistanceType(lP);
					noneKnnGlass.setMajorityMethod(majMethod);
					noneKnnGlass.buildClassifier(glassData);
					double currHyperAvgErr = noneKnnGlass.crossValidationError(noneKnnGlass.getTrainingInstances(), numOfFolds);

					if(currHyperAvgErr < minHyperAvgErr){
						hyperParmArrGlass[0] = K;
						hyperParmArrGlass[1] = lP;
						hyperParmArrGlass[2] = majMethod;
						minHyperAvgErr = currHyperAvgErr;
					}
				}
			}
		}

		if(hyperParmArrGlass[2] == 0)
			majorityMethod = "weighted";
		else
			majorityMethod = "uniform";
		System.out.println("Cross validation error with K = " + hyperParmArrGlass[0] + ", p = " + hyperParmArrGlass[1] 
				+ ", majority function  = " + majorityMethod + " for glass data is: "+minHyperAvgErr);

		// Cancer data set cross Validation error.

		// Choose Hyper Parameters by using cross validation
		numOfFolds = 10;
		minK = 1;
		maxK = 20;
		minHyperAvgErr = 1;
		majorityMethod="";

		Knn noneKnnCancer = new Knn();
		noneKnnCancer.setEditMode(EditMode.None);

		// Performing cross validation.
		for (int K = minK; K <= maxK; K++) {
			for (int lP = 0; lP <= 3; lP++) {
				for (int majMethod = 0; majMethod <= 1; majMethod++) {
					noneKnnCancer.setK(K);
					noneKnnCancer.setLPDistanceType(lP);
					noneKnnCancer.setMajorityMethod(majMethod);
					noneKnnCancer.buildClassifier(cancerData);
					double currHyperAvgErr = noneKnnCancer.crossValidationError(noneKnnCancer.getTrainingInstances(), numOfFolds);

					if(currHyperAvgErr < minHyperAvgErr){
						hyperParmArrCancer[0] = K;
						hyperParmArrCancer[1] = lP;
						hyperParmArrCancer[2] = majMethod;
						minHyperAvgErr = currHyperAvgErr;
					}
				}
			}
		}

		if(hyperParmArrCancer[2] == 0)
			majorityMethod = "weighted";
		else
			majorityMethod = "uniform";
		System.out.println("Cross validation error with K = " + hyperParmArrCancer[0] + ", p = " + hyperParmArrCancer[1] 
				+ ", majority function  = " + majorityMethod + " for cancer data is: "+minHyperAvgErr);

		noneKnnCancer.setK((int) hyperParmArrCancer[0]);
		noneKnnCancer.setLPDistanceType((int) hyperParmArrCancer[1]);
		noneKnnCancer.setMajorityMethod((int) hyperParmArrCancer[2]);
		noneKnnCancer.buildClassifier(cancerData);
		double[] precisionRecallArr = noneKnnCancer.calcConfusion(cancerData);
		System.out.println("The average Precision for the cancer dataset is: " + precisionRecallArr[0]);
		System.out.println("The average Recall for the cancer dataset is: "+precisionRecallArr[1]);


		// Part 2

		int[] numberOfFolds = new int[5];
		numberOfFolds[0] = glassData.size();
		numberOfFolds[1] = 50;
		numberOfFolds[2] = 10;
		numberOfFolds[3] = 5;
		numberOfFolds[4] = 3;

		for(int i=0; i<5;i++)
		{
			long startTime = System.nanoTime();
			noneKnnGlass.avarageFoldClassifyTime = 0;
			noneKnnGlass = new Knn();
			noneKnnGlass.setEditMode(EditMode.None);
			noneKnnGlass.setK((int) hyperParmArrGlass[0]);
			noneKnnGlass.setLPDistanceType((int) hyperParmArrGlass[1]);
			noneKnnGlass.setMajorityMethod((int) hyperParmArrGlass[2]);
			noneKnnGlass.buildClassifier(glassData);
			numOfFolds = numberOfFolds[i];
			Instances crossValidationInstances = noneKnnGlass.getTrainingInstances();
			double currHyperAvgErr = noneKnnGlass.crossValidationError(crossValidationInstances, numberOfFolds[i]);
			long totalTime = System.nanoTime() - startTime;

			System.out.println("----------------------------");
			System.out.println("Results for "+numOfFolds+ " folds:");
			System.out.println("----------------------------");
			System.out.println("Cross validation error of None-Edited knn on glass dataset is "+currHyperAvgErr+" and the average elapsed time is "+(noneKnnGlass.avarageFoldClassifyTime / numOfFolds));
			System.out.println("The total elapsed time is: "+totalTime);
			System.out.println("The total number of instances used in the classification phase is: "+((crossValidationInstances.size()/numOfFolds)*(numOfFolds-1)*numOfFolds));

			// Forwards Classifier

			startTime = System.nanoTime();
			Knn forwardsKnnGlass = new Knn();
			forwardsKnnGlass.avarageFoldClassifyTime = 0;
			forwardsKnnGlass.setEditMode(EditMode.Forwards);
			forwardsKnnGlass.setK((int) hyperParmArrGlass[0]);
			forwardsKnnGlass.setLPDistanceType((int) hyperParmArrGlass[1]);
			forwardsKnnGlass.setMajorityMethod((int) hyperParmArrGlass[2]);
			forwardsKnnGlass.buildClassifier(glassData);
			numOfFolds = numberOfFolds[i];
			crossValidationInstances = forwardsKnnGlass.getTrainingInstances();
			currHyperAvgErr = forwardsKnnGlass.crossValidationError(crossValidationInstances, numberOfFolds[i]);
			totalTime = System.nanoTime() - startTime;


			System.out.println("Cross validation error of Forwards-Edited knn on glass dataset is "+currHyperAvgErr+" and the average elapsed time is "+(noneKnnGlass.avarageFoldClassifyTime / numOfFolds));
			System.out.println("The total elapsed time is: "+totalTime);
			System.out.println("The total number of instances used in the classification phase is: "+((crossValidationInstances.size()/numOfFolds)*(numOfFolds-1)*numOfFolds));

			// Backwards Knn

			startTime = System.nanoTime();
			Knn backwardsKnn = new Knn();
			backwardsKnn.avarageFoldClassifyTime = 0;
			backwardsKnn.setEditMode(EditMode.Backwards);
			backwardsKnn.setK((int) hyperParmArrGlass[0]);
			backwardsKnn.setLPDistanceType((int) hyperParmArrGlass[1]);
			backwardsKnn.setMajorityMethod((int) hyperParmArrGlass[2]);
			backwardsKnn.buildClassifier(glassData);
			numOfFolds = numberOfFolds[i];
			crossValidationInstances = backwardsKnn.getTrainingInstances();
			currHyperAvgErr = backwardsKnn.crossValidationError(crossValidationInstances, numberOfFolds[i]);
			totalTime = System.nanoTime() - startTime;


			System.out.println("Cross validation error of Backwards-Edited knn on glass dataset is "+currHyperAvgErr+" and the average elapsed time is "+(noneKnnGlass.avarageFoldClassifyTime / numOfFolds));
			System.out.println("The total elapsed time is: "+totalTime);
			System.out.println("The total number of instances used in the classification phase is: "+((crossValidationInstances.size()/numOfFolds)*(numOfFolds-1)*numOfFolds));

		}
	}
}
