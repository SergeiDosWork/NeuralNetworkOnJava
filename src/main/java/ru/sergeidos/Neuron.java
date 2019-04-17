package ru.sergeidos;

import java.util.ArrayList;
import java.util.HashMap;

public class Neuron {

    static int counter = 0;
    final public int id;

    final double bias = -1; // Смешение
    Connection biasConnection; // Нейрон смещения
    double output;

    ArrayList<Connection> inConnectionList = new ArrayList<Connection>();
    HashMap<Integer,Connection> connectionLookup = new HashMap<Integer,Connection>();

    public Neuron(){
        id = counter;
        counter++;
    }

    /**
     * Вычисление выходного значения нейрона
     */
    public void calculateOutput(){
        double summ = 0;
        for(Connection connection : inConnectionList){
            Neuron leftNeuron = connection.getFromNeuron();
            double weight = connection.getWeight();
            double a = leftNeuron.getOutput();

            summ = summ + (weight*a);
        }
        summ = summ + (biasConnection.getWeight()*bias);

        output = activationFunction(summ);
    }


    double activationFunction(double x) {
        return sigmoid(x);
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 +  (Math.exp(-x)));
    }

    public void addInConnection(ArrayList<Neuron> inNeurons){
        for(Neuron neuron: inNeurons){
            Connection connection = new Connection(neuron,this);
            inConnectionList.add(connection);
            connectionLookup.put(neuron.id, connection);
        }
    }

    public Connection getConnection(int neuronIndex){
        return connectionLookup.get(neuronIndex);
    }

    public void addInConnection(Connection connection){
        inConnectionList.add(connection);
    }
    public void addBiasConnection(Neuron neuron){
        Connection connection = new Connection(neuron,this);
        biasConnection = connection;
        inConnectionList.add(connection);
    }
    public ArrayList<Connection> getAllInConnections(){
        return inConnectionList;
    }

    public double getBias() {
        return bias;
    }
    public double getOutput() {
        return output;
    }
    public void setOutput(double output){
        this.output = output;
    }
}