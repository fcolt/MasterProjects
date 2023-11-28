import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.io.FileFilter;
import java.util.Set;

public class AcceptedFileTypes implements FileFilter {
    private static final Set<String> ACCEPTED_FILETYPES = Set.of("docx", "doc", "pdf", "txt");
    public boolean accept(File pathname) {
        return ACCEPTED_FILETYPES.contains(FilenameUtils.getExtension(pathname.getName().toLowerCase()));
    }
}