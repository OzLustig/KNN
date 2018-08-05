package HomeWork4;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;



public class Knn implements Classifier {
	
	public enum EditMode {None, Forwards, Backwards};
	
	private EditMode m_editMode = EditMode.None;
	private Instances m_trainingInstances;
	private int m_K = 4;
	// infinity = 0
	public int m_LPDistanceType;
	public double m_AvgError;
	public int m_MajorityMethod;
	public long avarageFoldClassifyTime = 0;
	
	public class InstanceDistancePair implements Comparable<InstanceDistancePair>{
		public Instance instance;
		public double distance;
		
		//constructr
		public InstanceDistancePair(Instance instance, double distance){
			this.instance = instance;
			this.distance = distance;
		}
		
		// TODO: Idan, add comments, not clear what each of the return values mean
		@Override
		public int compareTo(InstanceDistancePair IDPair) {
			
			int result;
			if(this.distance < IDPair.distance){
				result = -1;
			}
			else if(this.distance > IDPair.distance){
				result = 1;
			}
			else{
				result = 0;
			}
			return result;
		}
	}
	
	public EditMode getEditMode() {
		return m_editMode;
	}

	public void setEditMode(EditMode editMode) {
		m_editMode = editMode;
	}

	public Instances getTrainingInstances() {
		return m_trainingInstances;
	}

	public void setTrainingInstances(Instances m_trainingInstances) {
		this.m_trainingInstances = m_trainingInstances;
	}

	public int getK() {
		return m_K;
	}

	public void setK(int m_K) {
		this.m_K = m_K;
	}

	public int getLPDistanceType() {
		return m_LPDistanceType;
	}

	public void setLPDistanceType(int m_LPDistanceType) {
		this.m_LPDistanceType = m_LPDistanceType;
	}

	public double getAvgError() {
		return m_AvgError;
	}

	public void setAvgError(double m_AvgError) {
		this.m_AvgError = m_AvgError;
	}

	public int getMajorityMethod() {
		return m_MajorityMethod;
	}

	public void setMajorityMethod(int m_MajorityMethod) {
		this.m_MajorityMethod = m_MajorityMethod;
	}

	
	

	

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		switch (m_editMode) {
		case None:
			noEdit(arg0);
			break;
		case Forwards:
			editedForward(arg0);
			break;
		case Backwards:
			editedBackward(arg0);
			break;
		default:
			noEdit(arg0);
			break;
		}
	}

	@Override
	public double classifyInstance(Instance instance) {

		double classification;
		ArrayList<InstanceDistancePair> IDPairArr = findNearestNeighbors(this.m_trainingInstances, instance);


		int numOfClassValues =  m_trainingInstances.numClasses();
		double[] classValuesArr = new double[numOfClassValues];
		if(this.m_MajorityMethod == 0)
			classification = getWeightedClassVoteResult(IDPairArr, classValuesArr);
		else
			classification = getClassVoteResult(IDPairArr, classValuesArr);
		
		return classification;
	}

	private void editedForward(Instances instances) 
	{
		Random random = new Random();
		instances.randomize(random);
		m_trainingInstances = new Instances(instances, 0, 1);
		for (Instance instance : instances) 
		{
			if(classifyInstance(instance) != instance.classValue())
			{
				m_trainingInstances.add(instance);
			}
		}
	}

	private void editedBackward(Instances instances) 
	{
		Random random = new Random();
		instances.randomize(random);
		m_trainingInstances = new Instances(instances);
		for( Instance instance : instances)
		{
			m_trainingInstances.remove(instance);
			if(classifyInstance(instance) != instance.classValue())
			{
				m_trainingInstances.add(instance);
			}
		}
	}

	private void noEdit(Instances instances) {
		m_trainingInstances = new Instances(instances);
	}
	
	
	
	/*********************************************************************
	 * 					Methods added by us!!!
	 *********************************************************************/
	
	/*
	 * 
	 * Calculate the Precision & Recall on a given instances set
	 *
	 * @param instances (a given instances set)
	 * @return double[] (double array of size 2. First index for Precision and the second for Recall)
	 */
	public double[] calcConfusion(Instances instances)
	{
		double[] confusionArray = new double[2];
		int truePositiveCount = 0, positiveCount = 0, falseNegativeCount = 0;
		for (Instance instance : instances)
		{
			double instanceClassValue = classifyInstance(instance);
			// Calculate Precision
			if( (instance.classValue() == 0) && (instanceClassValue == 0))
			{
				truePositiveCount++;
			}
			if(instanceClassValue == 0)
			{
				positiveCount++;
			}
			
			// Calculate recall
			if( (instanceClassValue == 1) && (instance.classValue() == 0))
			{
				falseNegativeCount++;
			}
			
		}
		confusionArray[0] = (double) truePositiveCount / positiveCount;
		confusionArray[1] = (double) truePositiveCount / (truePositiveCount + falseNegativeCount);
		return confusionArray;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}
	/**
	 *  calcAvgError
	 * @param dataSet (Instances)
	 * @return Average error (double)
	 */
	public double calcAvgError(Instances dataSet){
		
        int errorCounter = 0;
        double classification;
        for (Instance instance : dataSet) {
            classification = classifyInstance(instance);
            if (instance.classValue() != classification) {
                errorCounter++;
            }
        }
        return (double) errorCounter / (double) dataSet.size();
    }
	
	/**
	 * getClassVoteResult- Calculate the majority class of the neighbors
	 * @param KNearestNeighbors (Instances)- a set of K nearest neighbors
	 * @return result(double )
	 */
	public double getClassVoteResult(ArrayList<InstanceDistancePair> IDPairArr, double[] classValuesArr){

		for (InstanceDistancePair pair : IDPairArr) {
			int currInstanceClaseValueIndex = (int) pair.instance.classValue();
			classValuesArr[currInstanceClaseValueIndex]++;
		}

		double resultClass = -1;

		double maximumCounts = -1;
		for (int i = 0; i < classValuesArr.length; i++) {
			if( maximumCounts < classValuesArr[i]){
				maximumCounts = classValuesArr[i];
				resultClass = i;
			}
		}
		return resultClass;
	}
	
	/**
	 * Calculate the weighted majority class of the neighbors. In this method the class vote is normalized by the distance from the instance being classified. 
	 *
	 * @param KNearestNeighbors the k nearest neighbors
	 * @return the weighted class vote result
	 */
	public double getWeightedClassVoteResult(ArrayList<InstanceDistancePair> IDPairArr, double[] classValuesArr)
	{
		for (InstanceDistancePair pair: IDPairArr) 
		{
			int currInstanceClaseValueIndex = (int) pair.instance.classValue();
			classValuesArr[currInstanceClaseValueIndex] += (1 / Math.pow(pair.distance,2));
		}
		double resultClass = -1;
		double maximumCounts = -1;
		for (int i = 0; i < classValuesArr.length; i++) {
			if( maximumCounts < classValuesArr[i]){
				maximumCounts = classValuesArr[i];
				resultClass = i;
			}
		}
		return resultClass;
		}

		
	/**
	 * LPDistance finds the l-p distance between the two instances
	 * @param instance1 (Instance)
	 * @param instance2 (Instance)
	 * @return LPDistance (double)
	 */
	public double LPDistance(Instance instance1, Instance instance2){
		double result = 0;
		int lPType = this.m_LPDistanceType;
		
		if( lPType == 0)
			result = lInfinityDistance(instance1, instance2);
		else
			result = distance(instance1,instance2);
		
		return result;
	}
	
	/**
	 * lInfinityDistance finds the l infinity distance between the two instances
	 * @param instance1
	 * @param instance2
	 * @return maxAbsuluteDiference (double)
	 */
	public double lInfinityDistance(Instance instance1, Instance instance2){
		double sigma = 0;
		int dimension = instance1.numAttributes() - 1;
		double maxAbsuluteDiference = 0;
		// calculate the p sigma
		for (int l = 1; l < dimension; l++) {
			
			double lDimensionDifference = (instance1.value(l) - instance2.value(l));
			
			if(lDimensionDifference > maxAbsuluteDiference){
				maxAbsuluteDiference = lDimensionDifference;
			}
		}
		return maxAbsuluteDiference;
	}
	
	/**
	 * distance - finds the l-p distance (not infinity) 
	 * between the two instances according to the KNN LPDistanceType
	 * @param instance1
	 * @param instance2
	 * @return lPDistance (double)
	 */
	public double distance(Instance instance1, Instance instance2){
		int p = this.m_LPDistanceType;
		double sigma = 0;
		int dimension = instance1.numAttributes() - 1;
		// calculate the p sigma
		for (int l = 1; l < dimension; l++) {
			sigma += Math.pow((instance1.value(l) - instance2.value(l)), p);
		}
		double lPDistance = Math.pow(sigma, (double) 1/p);
		return lPDistance;
	}

	// TODO: Idan, fix the return value, it's not clear.
	/**
	 *  findNearestNeighbors - Find the K nearest neighbors for the instance being classified.
	 * @param instances (the instances to find the K nearest neighbors from)
	 * @param classifiedInstance (the instance whose being classified
	 * @return  KNNPairArr
	 */
	public ArrayList<InstanceDistancePair> findNearestNeighbors(Instances instances, Instance classifiedInstance){
		double maxDistanceInArr = 0, currentDistance;
		int instancesCounter = 0;
		ArrayList<InstanceDistancePair> IDPairArr = new ArrayList<InstanceDistancePair>();
		for (Instance instance : instances) {
			//if there are less then K instances in the KNNPairArr, add them to the K nearest neighbors ArrayList
			if(instancesCounter < this.m_K){
				currentDistance = LPDistance(classifiedInstance, instance);
				InstanceDistancePair currentPair = new InstanceDistancePair(instance, currentDistance);
				IDPairArr.add(currentPair);
				// Sort the result ArrayList in ascending order after adding a new element
				Collections.sort(IDPairArr);
				instancesCounter++;
			}
			//else if there are K instances in the KNNPairArr
			else{
				currentDistance = LPDistance(classifiedInstance, instance);
				InstanceDistancePair currPair = new InstanceDistancePair(instance, currentDistance);
				maxDistanceInArr = IDPairArr.get(this.m_K - 1).distance;
				
				//if the current instance distance is less the the maxDistanceInArr 
				//delete it, add the currPair and sort
				if(currPair.distance < maxDistanceInArr){
					IDPairArr.remove(this.m_K - 1);
					IDPairArr.add(currPair);
					Collections.sort(IDPairArr);
				}			
			}
		}
		return IDPairArr;
	 }

	public double crossValidationError(Instances instances, int numOfFolds){
		
		/*Shuffle instances*/
		Random random = new Random();
		instances.randomize(random);
		double sum = 0;
		if (numOfFolds > instances.size())
		{
			numOfFolds = instances.size();
		}
		
		Instances[] foldedInstances = foldCrossValidation(instances, numOfFolds);
		
		/*for every fold_i, divide to validation set and training set and then get the AvgErr*/
		for (int fold_i = 0; fold_i < numOfFolds; fold_i++) 
		{
			long currentStartTime = System.nanoTime();
			/*get fold cross validation division between validationSet and trainnigSet*/
			Instances validationSet = foldedInstances[fold_i];
			Instances trainnigSet = new Instances(instances, 0, 0);
			
			/*unify numOfFolds-1 sets to be trainingSet*/
			for (int i = 0; i < foldedInstances.length; i++) {
				if(i != fold_i){
					trainnigSet.addAll(foldedInstances[i]);
				}
			}
			
			this.m_trainingInstances = trainnigSet;
			double fold_i_AvgErr = calcAvgError(validationSet);
			sum += fold_i_AvgErr;
			avarageFoldClassifyTime+=(System.nanoTime() - currentStartTime);
		}
		
		return  sum / numOfFolds;
	}
	/**
	 * foldCrossValidatio - return a pair with validationSet and trainnigSet
	 * @param instances
	 * @param numOfFolds
	 * @return foldedInstances (Instances[]) 
	 */
	public Instances[] foldCrossValidation(Instances instances, int numOfFolds){
		//devide to 'numOfFolds' sets of Instances
		Instances[]  foldedInstances = new Instances[numOfFolds];
		
		/*innitialize all folds to be an empty Intances Objext*/
		for (int i = 0; i < foldedInstances.length; i++) {
			foldedInstances[i] = new Instances(instances, 0, 0);
		}
		
		// Add instances to each fold uniformly.
		
		for (int i = 0; i < instances.numInstances();i++) {
			
			foldedInstances[i % numOfFolds].add(instances.get(i));
		}

		return foldedInstances;
	}

}
