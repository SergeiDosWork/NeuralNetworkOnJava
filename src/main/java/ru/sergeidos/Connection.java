package ru.sergeidos;

public class Connection {
    static int counter = 0;
    final public int id;
    final Neuron fromNeuron;
    final Neuron toNeuron;
    double weight = 0;
    double prevDeltaWeight = 0;
    double deltaWeight = 0;

    public Connection(Neuron fromNeuron, Neuron toNeuron) {
        this.fromNeuron = fromNeuron;
        this.toNeuron = toNeuron;
        id = counter;
        counter++;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public void setDeltaWeight(double weight) {
        prevDeltaWeight = deltaWeight;
        deltaWeight = weight;
    }

    public double getPrevDeltaWeight() {
        return prevDeltaWeight;
    }

    public Neuron getFromNeuron() {
        return fromNeuron;
    }

    public Neuron getToNeuron() {
        return toNeuron;
    }
}