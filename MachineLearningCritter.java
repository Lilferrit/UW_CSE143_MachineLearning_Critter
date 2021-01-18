import java.awt.*;
import java.lang.*;
import java.util.*;

public class MachineLearningCritter extends Critter {
    private NeuralNetwork brain;
    private int hops;
    private static int pop;
    private int id;
    private final double MAXWEIGHT = 1.4;

    static {
        initStatic();
    }

    public Wheatley_GavinStraub() {
        hops = 0;
        pop++;
        id = pop;
        brain = new NeuralNetwork(10, 8);
    }


    public static void initStatic() {

    }

    /**
     * Grabs input from the Neural Network if the critter
     * is in open space. Otherwise turns or infects if another
     * critter is in front. The neural network class can be found
     * on line 188
     */
    public Action getMove(CritterInfo info) {
        if (info.getFront() == Neighbor.OTHER) {
            return Action.INFECT;
        } else if (info.getFront() == Neighbor.EMPTY) {
            return getSmartAction(info);
        } else {
            hops = 0;
            return getTurn(info);
        }
    }

    /**
     * Returns the blue of Wheatley's eye from
     * portal.
     */
    public Color getColor() {
        return new Color(0,89,216);
    }

    /**
     * Returns the critters unique ID, used for training
     */
    public String toString() {
        return "" + id;
    }

    /**
     * Grabs input from the Neural network and converts it
     * into an action
     */
    private Action getSmartAction(CritterInfo info) {
        double[] situation = getSituation(info);
        double outPut = brain.fire(situation);

        if (outPut <= .3333333333) {
            hops++;
            return Action.HOP;
        } else if (outPut <= .66666) {
            hops = 0;
            return Action.LEFT;
        } else {
            hops = 0;
            return Action.RIGHT;
        }
    }

    /**
     * Will cause critter to turn either left
     * or right depending on Neural Network input
     */
    private Action getTurn(CritterInfo info) {
        double[] situation = getSituation(info);
        double outPut = brain.fire(situation);

        if (outPut <= 0.5 ) {
            return Action.LEFT;
        } else {
            return Action.RIGHT;
        }
    }

    /**
     * Encodes various info methods into numbers that can be passed
     * into the neural network
     */
    private double[] getSituation(CritterInfo info) {
        double[] situation = new double[10];
        Direction[] directions = {Direction.NORTH, Direction.EAST,
                Direction.SOUTH, Direction.EAST};
        Neighbor[] neighbors = {Neighbor.WALL, Neighbor.EMPTY, Neighbor.SAME};

        situation[0] = encodeSituation(directions, info.getDirection());
        situation[1] = encodeSituation(neighbors, info.getFront());
        situation[2] = encodeSituation(neighbors, info.getBack());
        situation[3] = encodeSituation(neighbors, info.getLeft());
        situation[4] = encodeSituation(neighbors, info.getRight());
        situation[5] = encodeSituation(info.frontThreat());
        situation[6] = encodeSituation(info.backThreat());
        situation[7] = encodeSituation(info.leftThreat());
        situation[8] = encodeSituation(info.rightThreat());
        situation[9] = hops;

        return situation;
    }

    /**
     * Below are various methods for encoding info methods into usefull input
     * for the neural network.
     */
    private double encodeSituation(Direction[] directions, Direction current) {
        final double increment = 1/directions.length;
        double encoded = 0;
        double incremented = 0;

        for (int i = 0; i < directions.length; i++) {
            incremented += increment;
            if (directions[i] == current) {
                encoded = incremented;
            }
        }
        return encoded;
    }

    private double encodeSituation(Neighbor[] neighbors, Neighbor current) {
        final double increment = 1/neighbors.length;
        double encoded = 0;
        double incremented = 0;

        for (int i = 0; i < neighbors.length; i++) {
            incremented += increment;
            if (neighbors[i] == current) {
                encoded = incremented;
            }
        }
        return encoded;
    }

    private double encodeSituation(Boolean isThreat) {
        if (isThreat) {
            return 1;
        } else {
            return 0;
        }
    }

    private Action interpret(double output, Action[] actions) {
        double interval = 1/ actions.length;
        Action out = Action.LEFT;

        for (int i = 0; i < actions.length; i++) {
            if (output <= interval*(i + 1)) {
                out = actions[i];
            }
        }
        return out;
    }

    class NeuralNetwork {
        double[][][] hiddenWeights;
        double[][] inputWeights;
        double[] outputWeights;
        int input;
        int hidden;

        /**
         * Creates Matrices for the three kinds of wieghts, and
         * initializes them to a random value.
         */
        public NeuralNetwork(int input, int hidden) {
            Random rand = new Random();
            hiddenWeights = new double[hidden - 1][hidden][hidden];
            inputWeights = new double[hidden][input];
            outputWeights = new double[hidden];
            this.hidden = hidden;
            this.input = input;

            //Initialize hidden weights with random values
            //and initialize neuron
            for (int l = 0; l < hidden - 1; l++) {
                for (int r = 0; r < hidden; r++) {
                    for (int w = 0; w < hidden; w++) {
                        hiddenWeights[l][r][w] = rand.nextDouble()*MAXWEIGHT;
                    }
                }
            }

            //Initialize input weights with random values
            for (int r = 0; r < hidden; r++) {
                for (int w = 0; w < input; w++) {
                    inputWeights[r][w] = rand.nextDouble()*MAXWEIGHT;
                }
            }

            //Initialize output weights with random values
            for (int w = 0; w < hidden; w++) {
                outputWeights[w] = rand.nextDouble();
            }
        }

        /**
         * Returns the output of one Neuron
         */
        private double getOutput(double[] weights, double[] input) {
            if (weights.length != input.length) {
                throw new IllegalArgumentException("Input and weights arrays do" +
                        " not have the same length");
            }

            double cumSum = 0;

            for (int w = 0; w < weights.length; w++) {
                cumSum += weights[w] * input[w];
            }

            return sigmoid(cumSum);
        }

        /**
         * Always returns a value in between zero
         * and one.
         */
        private double sigmoid(double input) {
            return 1 / (1 + Math.exp(input));
        }

        /**
         * Based on an array of input numbers, will
         * propagate the input array through the neural
         * network and return a value in between zero
         * and one.
         */
        public double fire(double[] inputItems) {
            double[] prev = new double[hidden];
            double[] next = new double[hidden];

            //Start first layer
            for (int r = 0; r < hidden; r++) {
                prev[r] = getOutput(inputWeights[r], inputItems);
            }

            //Iterate over hidden layers
            for (int l = 1; l < hidden - 1; l++) {
                for (int r = 0; r < hidden; r++) {
                    next[r] = getOutput(hiddenWeights[l][r], prev);
                    prev = next;
                }
            }

            //get Final output
            return getOutput(outputWeights, next);
        }
    }
}
