import org.apache.lucene.queryparser.classic.ParseException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import org.apache.commons.cli.*;

public class Main {
    public static void main(String[] args) throws IOException, ParseException {
        Options options = new Options();

        Option docDir = new Option("p", "docDir", true, "The path of the documents to be analyzed");
        docDir.setRequired(true);
        options.addOption(docDir);

        Option indexDir = new Option("i", "indexDir", true, "The path in which to store indices");
        indexDir.setRequired(true);
        options.addOption(indexDir);

        CommandLineParser cmdLineParser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd = null;
        try {
            cmd = cmdLineParser.parse(options, args);
        } catch (org.apache.commons.cli.ParseException e) {
            formatter.printHelp("utility-name", options);
        }

        Analyzer analyzer;
        analyzer = new Analyzer(cmd.getOptionValue("docDir"), cmd.getOptionValue("indexDir"));

        while (true) {
            System.out.println("Ce termen cautam?");
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String cautare = reader.readLine();

            if (cautare.length() > 1) {
                analyzer.search(cautare.toLowerCase().trim());
            } else {
                System.out.println("Cel putin 2 caractere");
            }
        }
    }
}