//! Maps HIR entities to filesystem paths and line numbers.

use std::path::{Path, PathBuf};

use ra_ap_base_db::EditionedFileId;
use ra_ap_ide_db::RootDatabase;
use ra_ap_ide_db::LineIndexDatabase;
use ra_ap_syntax::TextSize;
use ra_ap_vfs::Vfs;

/// Resolves HIR file IDs to relative paths and byte offsets to line numbers.
pub struct SourceMapper {
    project_root: PathBuf,
}

impl SourceMapper {
    pub fn new(project_root: &Path) -> Self {
        Self {
            project_root: project_root.to_path_buf(),
        }
    }

    /// Convert an EditionedFileId to a project-relative path string.
    pub fn file_path(&self, vfs: &Vfs, db: &RootDatabase, file_id: EditionedFileId) -> Option<String> {
        let vfs_file_id = file_id.file_id(db);
        let vfs_path = vfs.file_path(vfs_file_id);
        let abs_path = vfs_path.as_path()?;
        let abs_buf: &Path = abs_path.as_ref();
        let relative = abs_buf
            .strip_prefix(&self.project_root)
            .unwrap_or(abs_buf);
        Some(relative.to_string_lossy().to_string())
    }

    /// Convert a byte offset within a file to a 1-based line number.
    pub fn line_number(&self, db: &RootDatabase, file_id: EditionedFileId, offset: TextSize) -> Option<u32> {
        // line_index takes vfs::FileId, not EditionedFileId.
        let vfs_file_id = file_id.file_id(db);
        let line_index = db.line_index(vfs_file_id);
        let line_col = line_index.line_col(offset);
        Some(line_col.line + 1) // 0-indexed → 1-indexed
    }
}
