package ru.sergeidos;

import java.text.DecimalFormat;
import java.util.*;

public class SegaNeytron {

    final Random rand = new Random();
    final ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
    final ArrayList<Neuron> hiddenLayer = new ArrayList<Neuron>();
    final ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
    final Neuron bias = new Neuron();
    final int[] layers;
    final int randomWeightMultiplier = 1;
    static final int byteCount = 20; // Количество бит в числе
    static final int learningCount = 100; // На скольки первых числах обучаться

    final double epsilon = 0.00000000001;

    final double learningRate = 0.9f; // Скорость обучения
    final double momentum = 0.7f; // Момент

    final HashMap<String, Double> weightUpdate = new HashMap<String, Double>();

    double inputs[][] ; // Входящие нейроны
    double expectedOutputs[][]; // Идеальный результат к которому будем стремиться
    double output[]; // Слой выходных нейронов

    double resultOutputs[][]; //

    {
        // Проинициализируем параметры
        inputs = new double[learningCount][];
        expectedOutputs = new double[learningCount][];
        resultOutputs = new double[learningCount][];

        for (int i=0; i<learningCount; i++) {
            resultOutputs[i] = new double[1];
            resultOutputs[i][0] = -1;

            inputs[i] = new double[byteCount];
            inputs[i] = toAr(i); // Будем раскладывать по массиву двоичное значение

            expectedOutputs[i] = new double[1];
            expectedOutputs[i][0] = i % 2; // Идеальное число для обучения, 1.0 - нечетное, 0.0 - четное

        }
    }

    public SegaNeytron(int inputCount, int hiddenCount, int outputCount) {
        this.layers = new int[] { inputCount, hiddenCount, outputCount };
        //        df = new DecimalFormat("#.0#");

        //  Создадим нейросеть со случайными значениями связей
        // Входящий слой
        for (int j = 0; j < inputCount; j++) {
            Neuron neuron = new Neuron();
            inputLayer.add(neuron);
        }
        // Скрытый слой
        for (int j = 0; j < hiddenCount; j++) {
            Neuron neuron = new Neuron();
            neuron.addInConnection(inputLayer);
            neuron.addBiasConnection(bias);
            hiddenLayer.add(neuron);
        }
        // Выходной слой
        for (int j = 0; j < outputCount; j++) {
            Neuron neuron = new Neuron();
            neuron.addInConnection(hiddenLayer);
            neuron.addBiasConnection(bias);
            outputLayer.add(neuron);
        }
        // Замутим случайные значения
        for (Neuron neuron : hiddenLayer) {
            ArrayList<Connection> connections = neuron.getAllInConnections();
            for (Connection conn : connections) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            }
        }
        for (Neuron neuron : outputLayer) {
            ArrayList<Connection> connections = neuron.getAllInConnections();
            for (Connection conn : connections) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            }
        }
    }

    // Переводит целое число в набор бит
    static double[] toAr(int a) {
        String maxAmpStr = Integer.toBinaryString(a);
        while (maxAmpStr.length()<byteCount) {
            maxAmpStr = "0"+maxAmpStr;
        }
        double[] arr = new double[byteCount];
        for (int i=0; i<byteCount; i++ ) {
            String v = maxAmpStr.substring(i,i+1);
            arr[i] = Double.valueOf(v);
        }
        return arr;
    }

    public static void main(String[] args) {
        // Замутим запуск обучения
        SegaNeytron segaNeytron = new SegaNeytron(byteCount, 2, 1);
        int maxRuns = 500000; // Максимальное количество эпох
        double minErrorCondition = 0.0001; // Минимальный размер допустимой среднеквадратичной ошибки
        if (segaNeytron.run(maxRuns, minErrorCondition)) {

            // Тест
            int err=0, ev=0, od=0;
            List<Integer> l = new LinkedList<Integer>();
            int cnt = (int) (Math.pow(2,byteCount)-1);
            for (int i=1; i<cnt; i++ ) {
                segaNeytron.setInput(toAr(i));
                segaNeytron.activate();
                double[] output = segaNeytron.getOutput();
                if ((output[0]<0.5 && i%2==0) || (output[0]>=0.5 && i%2==1)) {
                    //верно
                } else {
                    //не верно
//                    System.out.println("На числе "+i+ " сетка тупит ("+output[0]+","+i%2+")");
                    err++;
                    l.add(i);
                    if (i%2==0) {
                        ev++;
                    } else {
                        od++;
                    }

                }
            }
            System.out.println("При обучении на числах от 1 до "+learningCount+", при тестовом прогоне на числах от 1 до "+ cnt +" сеть ошибласть в "+err+ " случаях ("+ev+" четных, "+od+" нечетных)");
            if (l.size()>2) {
                System.out.println("Вот три примера таких чисел:");
                for (int i=0; i<=2; i++) {
                    int x = new Random().nextInt(l.size());
                    segaNeytron.setInput(toAr(l.get(x)));
                    segaNeytron.activate();
                    double[] output = segaNeytron.getOutput();
                    System.out.println("Число "+l.get(x)+" сеть посчитала "+(output[0]<0.5?"четным":"нечетным")+" (предсказание="+output[0]+") а на самом деле оно "+(l.get(x)%2==0?"четное":"нечетное")+" (идеал="+l.get(x)%2+")") ;
                }

                segaNeytron.setInput(toAr(l.get(0)));
                segaNeytron.activate();
                double[] output = segaNeytron.getOutput();
                System.out.println("Минимальное число "+l.get(0)+" сеть посчитала "+(output[0]<0.5?"четным":"нечетным")+" (предсказание="+output[0]+") а на самом деле оно "+(l.get(0)%2==0?"четное":"нечетное")+" (идеал="+l.get(0)%2+")") ;

                segaNeytron.setInput(toAr(l.get(l.size()-1)));
                segaNeytron.activate();
                output = segaNeytron.getOutput();
                System.out.println("Максимальное число "+l.get(l.size()-1)+" сеть посчитала "+(output[0]<0.5?"четным":"нечетным")+" (предсказание="+output[0]+") а на самом деле оно "+(l.get(l.size()-1)%2==0?"четное":"нечетное")+" (идеал="+l.get(l.size()-1)%2+")") ;
            }

            Scanner scanner = new Scanner(System.in);
            System.out.println("Введите число >0");
            int enter = scanner.nextInt();
            while (enter>0) {
                double[] inValue = toAr(enter);
                segaNeytron.setInput(inValue);
                segaNeytron.activate();
                double[] output = segaNeytron.getOutput();

                System.out.println("Наша нейронная сеть делает предсказание, что ваше число:");
                if (output[0]<0.5) {
                    System.out.println("Четное!!\n");
                } else {
                    System.out.println("Не четное!!11один\n");
                }
                System.out.println("Введите число >0");
                enter = scanner.nextInt();
            }
        }

    }

    // random
    double getRandom() {
        return randomWeightMultiplier * (rand.nextDouble() * 2 - 1); // -1 .. 1
    }

    public void setInput(double inputs[]) {
        for (int i = 0; i < inputLayer.size(); i++) {
            inputLayer.get(i).setOutput(inputs[i]);
        }
    }

    public double[] getOutput() {
        double[] outputs = new double[outputLayer.size()];
        for (int i = 0; i < outputLayer.size(); i++)
            outputs[i] = outputLayer.get(i).getOutput();
        return outputs;
    }

    public void activate() {
        for (Neuron n : hiddenLayer)
            n.calculateOutput();
        for (Neuron n : outputLayer)
            n.calculateOutput();
    }


    public void applyBackpropagation(double expectedOutput[]) {
        for (int i = 0; i < expectedOutput.length; i++) {
            double expected = expectedOutput[i];
            if (expected < 0 || expected > 1) {
                if (expected < 0)
                    expectedOutput[i] = 0 + epsilon;
                else
                    expectedOutput[i] = 1 - epsilon;
            }
        }

        int i = 0;
        for (Neuron neuron : outputLayer) {
            ArrayList<Connection> connections = neuron.getAllInConnections();
            for (Connection connection : connections) {
                double ak = neuron.getOutput();
                double ai = connection.fromNeuron.getOutput();
                double desiredOutput = expectedOutput[i];

                double partialDerivative = -ak * (1 - ak) * ai * (desiredOutput - ak); // Производная
                double deltaWeight = -learningRate * partialDerivative; // смещение веса
                double newWeight = connection.getWeight() + deltaWeight;
                connection.setDeltaWeight(deltaWeight);
                connection.setWeight(newWeight + momentum * connection.getPrevDeltaWeight());
            }
            i++;
        }

        // обновляем веса скрытого слоя
        for (Neuron n : hiddenLayer) {
            ArrayList<Connection> connections = n.getAllInConnections();
            for (Connection con : connections) {
                double aj = n.getOutput();
                double ai = con.fromNeuron.getOutput();
                double sumKoutputs = 0;
                int j = 0;
                for (Neuron neuron : outputLayer) {
                    double wjk = neuron.getConnection(n.id).getWeight();
                    double desiredOutput = (double) expectedOutput[j];
                    double ak = neuron.getOutput();
                    j++;
                    sumKoutputs = sumKoutputs + (-(desiredOutput - ak) * ak * (1 - ak) * wjk);
                }

                double partialDerivative = aj * (1 - aj) * ai * sumKoutputs;
                double deltaWeight = -learningRate * partialDerivative;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
            }
        }
    }

    boolean run(int maxEpochsCount, double minErrorLevel) {
        // Тренируем сеть пока не достигнем допустимый уровень ошибки minErrorLevel или maxEpochsCount количества эпох
        double errorLevel = 1;
        int i = 0;

        while (i < maxEpochsCount && errorLevel > minErrorLevel) {
            errorLevel = 0;
            for (int p = 0; p < inputs.length; p++) { // Перебираем тренировочный набор
                setInput(inputs[p]); // Заполняем значения из тренировочного набора
                activate();  // Крутите барабан!

                output = getOutput(); // Собираем значения на выходе
                resultOutputs[p] = output;

                // Считаем сумму среднеквадратичной ошибки отклонения от эталона
                for (int j = 0; j < expectedOutputs[p].length; j++) {
                    double err = Math.pow(output[j] - expectedOutputs[p][j], 2);
                    errorLevel += err;
                }

                // Корректируем значения методом обратного распространения
                applyBackpropagation(expectedOutputs[p]);
            }
            i++;
        }

        printResult();

        System.out.printf("Сумма среднеквадратичной ошибки %.15f \n", errorLevel);
        System.out.println("Эпоха " + i);
        if (i == maxEpochsCount) {
            System.out.println("Обучение завершилось неудачно");
            return false;
        } else {
            System.out.println("Обучение завершилось!");
            return true;
        }
    }

    void printResult()
    {
        System.out.println("Результат тренировки");
        for (int p = 0; p < inputs.length; p++) {
            System.out.print("На входе: ");
            for (int x = 0; x < layers[0]; x++) {
                System.out.printf("%.1f ", inputs[p][x] );
            }

            System.out.print("Ожидаем на выходе: ");
            for (int x = 0; x < layers[2]; x++) {
                System.out.printf("%.1f ", expectedOutputs[p][x]);
            }

            System.out.print("Наше значение: ");
            for (int x = 0; x < layers[2]; x++) {
                System.out.printf("%.15f ", resultOutputs[p][x]);
            }
            System.out.println();
        }
        System.out.println();
    }

}