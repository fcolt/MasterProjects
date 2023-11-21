import org.apache.lucene.queryparser.classic.ParseException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import org.apache.commons.cli.*;
import org.apache.tika.exception.TikaException;

public class Main {
    public static void main(String[] args) throws IOException, ParseException, TikaException {
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

        IndexSearcher indexSearcher = new IndexSearcher(cmd.getOptionValue("docDir"), cmd.getOptionValue("indexDir"));

        while (true) {
            System.out.println("Query: ");
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String query = reader.readLine();

            indexSearcher.search(query.toLowerCase().trim(), 10);
        }
    }
}