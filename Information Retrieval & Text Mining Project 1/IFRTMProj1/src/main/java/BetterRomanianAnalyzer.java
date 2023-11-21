import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.apache.lucene.analysis.miscellaneous.SetKeywordMarkerFilter;
import org.apache.lucene.analysis.ro.RomanianAnalyzer;
import org.apache.lucene.analysis.snowball.SnowballFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.tartarus.snowball.ext.RomanianStemmer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileReader;

public class BetterRomanianAnalyzer extends StopwordAnalyzerBase {
    private final CharArraySet stemExclusionSet;

    public static final String DEFAULT_STOPWORD_FILE = "stopwords.txt";
    public static final String EXTRA_STOPWORD_FILE = "ro_stopwords.txt";
    private static final String STOPWORDS_COMMENT = "#";

    public static CharArraySet getDefaultStopSet() {
        return DefaultSetHolder.DEFAULT_STOP_SET;
    }

    private static class DefaultSetHolder {
        static final CharArraySet DEFAULT_STOP_SET;

        static {
            try {
                DEFAULT_STOP_SET = loadStopwordSet(false, RomanianAnalyzer.class,
                        DEFAULT_STOPWORD_FILE, STOPWORDS_COMMENT);
            } catch (IOException ex) {
                throw new RuntimeException("Unable to load default stopword set");
            }
        }
    }

    public BetterRomanianAnalyzer() {
        this(DefaultSetHolder.DEFAULT_STOP_SET);
    }

    public BetterRomanianAnalyzer(CharArraySet stopwords) {
        this(stopwords, CharArraySet.EMPTY_SET);
    }

    public BetterRomanianAnalyzer(CharArraySet stopwords, CharArraySet stemExclusionSet) {
        super(stopwords);
        this.stemExclusionSet = CharArraySet.unmodifiableSet(CharArraySet.copy(stemExclusionSet));
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer source = new StandardTokenizer();
        CharArraySet stopWordSet;
        try {
            stopWordSet = loadAdditionalStopwords();
        } catch (IOException e) {
            stopWordSet = CharArraySet.copy(DefaultSetHolder.DEFAULT_STOP_SET);
        }

        TokenStream result = new LowerCaseFilter(source);
        result = new ASCIIFoldingFilter(result);
        result = new StopFilter(result, stopWordSet);
        if (!this.stemExclusionSet.isEmpty()) {
            result = new SetKeywordMarkerFilter((TokenStream) result, this.stemExclusionSet);
        }
        result = new SnowballFilter((TokenStream) result, new RomanianStemmer());
        return new TokenStreamComponents(source, result);
    }

    private CharArraySet loadAdditionalStopwords() throws IOException {
        CharArraySet additionalStopwords = CharArraySet.copy(DefaultSetHolder.DEFAULT_STOP_SET);
        BufferedReader reader = new BufferedReader(new FileReader(EXTRA_STOPWORD_FILE));
        String line = reader.readLine();

        while (line != null) {
            additionalStopwords.add(line.trim());
            line = reader.readLine();
        }

        reader.close();

        return additionalStopwords;
    }
}