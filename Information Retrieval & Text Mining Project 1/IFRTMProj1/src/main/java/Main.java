import org.apache.lucene.queryparser.classic.ParseException;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Objects;

import org.apache.commons.cli.*;
import org.apache.tika.exception.TikaException;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException, ParseException, TikaException {
        Options options = new Options();

        Option docDirOption = new Option("p", "docDir", true, "The path of the documents to be analyzed");
        docDirOption.setRequired(true);
        options.addOption(docDirOption);

        Option indexDirOption = new Option("i", "indexDir", true, "The path in which to store indices");
        indexDirOption.setRequired(true);
        options.addOption(indexDirOption);

        Option noOfHitsOption = new Option("h", "noOfHits", true, "The maximum number of hits of a query");
        noOfHitsOption.setRequired(false);
        options.addOption(noOfHitsOption);

        Option queryDirOption = new Option("q", "queryDir", true, "The directory from which to read queries");
        noOfHitsOption.setRequired(false);
        options.addOption(queryDirOption);

        CommandLineParser cmdLineParser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd = null;
        try {
            cmd = cmdLineParser.parse(options, args);
        } catch (org.apache.commons.cli.ParseException e) {
            formatter.printHelp("utility-name", options);
        }

        IndexSearcher indexSearcher = new IndexSearcher(cmd.getOptionValue("docDir"), cmd.getOptionValue("indexDir"));
        int noOfHits = Objects.isNull(cmd.getOptionValue("noOfHits"))
                ? indexSearcher.filesIndexed
                : Integer.parseInt(cmd.getOptionValue("noOfHits"));

        String queryDir = cmd.getOptionValue("queryDir");
        if (!Objects.isNull(queryDir)) {
            File queryFolder = new File(queryDir);
            File[] files = queryFolder.listFiles();
            if (Objects.isNull(files)) {
                System.out.println("No queries in folder.");
                return;
            }

            for (File file : files) {
                System.out.print("Showing results for " + file.getName() + ":\n");
                List<String> lines = Files.readAllLines(file.toPath(), StandardCharsets.UTF_8);
                for (String line : lines) {
                    if (line.isEmpty()) {
                        continue;
                    }
                    indexSearcher.search(line.toLowerCase().trim(), noOfHits);
                }
            }

            return;

        }

        while (true) {
            System.out.println("Query: ");
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String query = reader.readLine();

            indexSearcher.search(query.toLowerCase().trim(), noOfHits);
        }
    }
}