import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Class to perform NLP analysis
 *
 *  @author Mien Nguyen, completed functions as submission for PS-5
 *
 */

public class NLP {
    // files for training
    private BufferedReader sentencesTrain;
    private BufferedReader labelsTrain;

    // files for testing
    private BufferedReader sentencesTest;
    private BufferedReader labelsTest;

    // observations & transitions map filled by training
    private Map<String, Map<String, Double>> observations;
    private Map<String, Map<String, Double>> transitions;

    private double unobservedScore = -100.0; // score for unseen words

    /**
     * Trains model using a sentences file & labels file
     * @param sentences path to file with sentences
     * @param labeling path to file with labels
     */
    public void training(String sentences, String labeling) {
        try {
            observations = new HashMap<>();
            transitions = new HashMap<>();

            // open both files for reading
            sentencesTrain = new BufferedReader(new FileReader(sentences));
            labelsTrain = new BufferedReader(new FileReader(labeling));

            String sentence; // keeps track of the sentence we're currently reading from file
            String labels; // keeps track of the labels we're currently reading from file
            while ((sentence = sentencesTrain.readLine()) != null && (labels = labelsTrain.readLine()) != null) { // read by line until end of file
                sentence = sentence.toLowerCase(); // upper case shouldn't factor into labeling

                // each element in sentenceArr should have label at same index in labelsArr
                String[] sentenceArr = sentence.split(" ");
                String[] labelsArr = labels.split(" ");

                // build observations map
                for (int i=0; i<sentenceArr.length; i++) {
                    if (!observations.containsKey(labelsArr[i])) { // if we've never seen this label before, add it
                        observations.put(labelsArr[i], new HashMap<>());
                    }
                    if (observations.get(labelsArr[i]).containsKey(sentenceArr[i])) { // if we've seen this word in this pos, add 1 to frequency
                        observations.get(labelsArr[i]).put(sentenceArr[i], observations.get(labelsArr[i]).get(sentenceArr[i])+1.0);
                    }
                    else { // if we've never seen this word in this pos before, add it, its frequency should be 1 initially
                        observations.get(labelsArr[i]).put(sentenceArr[i], 1.0);
                    }
                }

                // build transitions map
                if (!transitions.containsKey("#")) { // add the hashtag signifying start of sentence (labeling doesn't have this so must add manually)
                    transitions.put("#", new HashMap<>());
                }
                if (transitions.get("#").containsKey(labelsArr[0])) { // if we've seen a sentence start with this label before, add 1 to frequency
                    transitions.get("#").put(labelsArr[0], transitions.get("#").get(labelsArr[0])+1.0);
                }
                else { // otherwise set initial frequency to 1
                    transitions.get("#").put(labelsArr[0], 1.0);
                }
                for (int i=0; i<labelsArr.length-1; i++) {
                    if (!transitions.containsKey(labelsArr[i])) { // if we've never seen something transition to another state from this label before, add it
                        transitions.put(labelsArr[i], new HashMap<>());
                    }
                    if (transitions.get(labelsArr[i]).containsKey(labelsArr[i+1])) { // if we've transitioned from this label to next label in labeling file, increment frequency
                        transitions.get(labelsArr[i]).put(labelsArr[i+1], transitions.get(labelsArr[i]).get(labelsArr[i+1])+1.0);
                    }
                    else { // otherwise set initial frequency to 1
                        transitions.get(labelsArr[i]).put(labelsArr[i+1], 1.0);
                    }
                }
            }

            // normalize frequencies in both maps
            for (String pos: observations.keySet()) {
                double sumObs = 0.0; // keeps track of sum of observations of every word in a pos category
                for (String word: observations.get(pos).keySet()) {
                    sumObs += observations.get(pos).get(word);
                    //System.out.println(word + " " + observations.get(pos).get(word));
                }

                // divide frequency of every word with that pos by sum observations
                for (String word: observations.get(pos).keySet()) {
                    observations.get(pos).put(word, Math.log(observations.get(pos).get(word)/sumObs));
                }
            }

            for (String pos: transitions.keySet()) {
                double sumTrans = 0.0; // keeps track of sum of transitions to next pos from current pos category
                for (String next: transitions.get(pos).keySet()) {
                    sumTrans += transitions.get(pos).get(next);
//                    System.out.println(pos + " " + next + " " + transitions.get(pos).get(next));
                }

                // divide frequency of every next from current pos by sum transitions
                for (String next: transitions.get(pos).keySet()) {
                    transitions.get(pos).put(next, Math.log(transitions.get(pos).get(next)/sumTrans));
                }
            }
//            System.out.println(observations);
//            System.out.println("-----");
//            System.out.println(transitions);
        } catch (IOException e) {
            e.printStackTrace();
        }
        finally {
            if (sentencesTrain != null) { // close train file with sentences if opened
                try {
                    sentencesTrain.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (labelsTrain != null) { // close train file with labels if opened
                try {
                    labelsTrain.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

    }

    /**
     * Predicts labels of a line passed in
     * @param line the sentence you want to get predictive labels for
     * @return list of labels for every space-delimited unit in the line passed in
     */
    public List<String> viterbi(String line) {
        List<String> tags = new ArrayList<>(); // build this up and return this
        List<Map<String, String>> backtrace = new ArrayList<>(); // key: next, val: curr

        String[] lineArr = line.split(" ");
        Set<String> currStates = new HashSet<>();
        Map<String, Double> currScores = new HashMap<>();

        // handle start of sentence, which isn't explicit in line
        currStates.add("#");
        currScores.put("#", 0.0);

        for (int i=0; i<lineArr.length; i++) {
            Set<String> nextStates = new HashSet<>();
            Map<String, Double> nextScores = new HashMap<>();
            backtrace.add(new HashMap<>()); // backtrace must have new entry for every word/unit in the line we pass in

            for (String currState: currStates) { // for every current state

                if (transitions.containsKey(currState)) { // safety check that in training we've seen transitions going from this state before

                    for (String nextState : transitions.get(currState).keySet()) { // look into what states we can transition to
                        nextStates.add(nextState);

                        // value of observation score depends on whether we've seen it
                        double obsScore;
                        if (!observations.get(nextState).containsKey(lineArr[i])) {
                            obsScore = unobservedScore; // if not then just really low value
                        } else {
                            obsScore = observations.get(nextState).get(lineArr[i]); // otherwise get the actual score we've assigned
                        }

                        double nextScore = currScores.get(currState) + transitions.get(currState).get(nextState) + obsScore; // calculate score for going to next state

                        if (!nextScores.containsKey(nextState) || (nextScores.containsKey(nextState) && nextScore > nextScores.get(nextState))) {
                            nextScores.put(nextState, nextScore); // update score associated with next state if not calculated before or if new score higher than old
                            // update backtrace accordingly
                            backtrace.get(backtrace.size()-1).remove(nextState);
                            backtrace.get(backtrace.size()-1).put(nextState, currState);
                        }
                    }
                }
            }
            // update before moving onto next word
            currStates = nextStates;
            currScores = nextScores;
        }
//        System.out.println(currStates);

        // make curr the tag with the highest score from currScores
        String curr = null;
        double highest = -10000000000.0; // dummy value
        for (String state: currScores.keySet()) {
            if (curr == null) {curr = state;}
            if (currScores.get(state) > highest) {
                highest = currScores.get(state);
                curr = state;
            }
        }

        Stack<String> tagBuilder = new Stack<>(); // just so the list is in right order later
        tagBuilder.push(curr); // remember to include 1st item
        for (int i=backtrace.size()-1; i>=0; i--) { // go backward
            Map<String, String> currMap = backtrace.get(i);
//            System.out.println(backtrace.get(i));
            // get the tag that comes before curr and add to stack
            String prev = currMap.get(curr);
            tagBuilder.push(prev);
            // update curr to prev to continue moving backward
            curr = prev;
        }

        // build our list in the right order
        while (!tagBuilder.isEmpty()) {
            String toAdd = tagBuilder.pop();
            if (!toAdd.equals("#")) { // care to not add hashtag
                tags.add(toAdd);
            }
        }

        return tags;
    }

    /**
     * Prompts user for a sentence then prints out the tags list for that sentence
     */
    public void consoleTest() {
        Scanner in = new Scanner(System.in);
        while (true) {
            System.out.println("Enter a sentence for a prediction or q to quit >");
            String sentence = in.nextLine();
            if (sentence.equals("q")) {break;} // option to quit
            System.out.println(viterbi(sentence)); // if no quit, print out predicted labels
        }
    }

    /**
     * Takes in 2 files, sentences and corresponding labels, carries out labeling with viterbi and prints out performance
     * @param sentences path to file with sentences
     * @param labeling path to file with corresponding labels
     */
    public void filesTest(String sentences, String labeling) {
        try {
            // for accuracy score calculation
            double total = 0.0;
            double correct = 0.0;

            // open both files for reading
            sentencesTest = new BufferedReader(new FileReader(sentences));
            labelsTest = new BufferedReader(new FileReader(labeling));

            String sentence;
            String labels;
            while ((sentence = sentencesTest.readLine()) != null && (labels = labelsTest.readLine()) != null) { // read by line until end of file
                sentence = sentence.toLowerCase(); // upper case shouldn't affect labeling
                List<String> prediction = viterbi(sentence); // get the prediction for current sentence
                String[] labelsArr = labels.split(" ");
                if (prediction.size() == labelsArr.length) { // safety check that the length of our predicted labels & actual labels are the same
                    total += labelsArr.length; // add to total labels predicted
                    for (int i=0; i<prediction.size(); i++) {
                        if (prediction.get(i).equals(labelsArr[i])) {
                            correct += 1.0; // add to correctly predicted labels if correctly predicted
                        }
                    }
                }
            }
            // print out performance
            System.out.println("Out of " + (int)total + " tags, the model got " + (int)correct + " right and " + (int)(total-correct) + " wrong.");
            System.out.println("The accuracy is: " + (correct/total));

        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (sentencesTest != null) { // close sentences file if opened
                try {
                    sentencesTest.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (labelsTest != null) { // close labels file if opened
                try {
                    labelsTest.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static void main(String[] args) {
        NLP model = new NLP();
//        model.training("texts/simple-train-sentences.txt", "texts/simple-train-tags.txt");
//        System.out.println(model.viterbi("we work for trains ."));
//        System.out.println(model.viterbi("we work for trains"));
//        System.out.println(model.viterbi("he trains the train ."));
//        test.consoleTest();

        // train then run on test data
        model.training("texts/brown-train-sentences.txt", "texts/brown-train-tags.txt");
        model.filesTest("texts/brown-test-sentences.txt", "texts/brown-test-tags.txt");
    }
}
