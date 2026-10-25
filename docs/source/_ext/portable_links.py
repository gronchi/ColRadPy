"""Relative links that work both in the Sphinx build and in GitHub's preview of the pages.

GitHub renders the .rst files with plain docutils, which shows Sphinx roles such as :doc: and
:download: as raw text. A page can instead use ordinary relative links, which GitHub follows:

    `theory manual <theory_manual.rst>`__
    `colradpy_paper.pdf <files/colradpy_paper.pdf>`__

This extension turns them into what the roles make in the build: a link to a .rst file of the
documentation becomes a :doc: cross-reference to the built page, and a link to another file of
the source tree becomes a :download: of that file. Links with a scheme or an anchor are left
alone.
"""

import os.path
import posixpath

from docutils import nodes
from sphinx import addnodes
from sphinx.transforms import SphinxTransform


class PortableLinks(SphinxTransform):
    """Relative links to .rst files become :doc: references, to other files :download:s."""

    # after docutils has resolved the hyperlinks (priorities 440-660) and before Sphinx emits
    # doctree-read (880), whose collector registers the files of download references
    default_priority = 700

    def apply(self, **kwargs):
        docname = self.env.docname
        for ref in list(self.document.findall(nodes.reference)):
            uri = ref.get('refuri', '')
            if not uri or '://' in uri or uri.startswith(('#', '/', 'mailto:')) or '#' in uri:
                continue
            text = ref.astext()
            if uri.endswith('.rst'):
                target = uri[:-len('.rst')]
                if posixpath.normpath(posixpath.join(posixpath.dirname(docname), target)) \
                        not in self.env.found_docs:
                    continue
                new = addnodes.pending_xref('', nodes.inline(text, text, classes=['xref', 'std', 'std-doc']),
                                            refdomain='std', reftype='doc', reftarget=target,
                                            refexplicit=True, refwarn=True, refdoc=docname)
            else:
                _, path = self.env.relfn2path(uri, docname)
                if not os.path.isfile(path):
                    continue
                new = addnodes.download_reference('', '', nodes.literal(text, text, classes=['xref', 'download']),
                                                  reftarget=uri, refdomain='', reftype='download',
                                                  refexplicit=True, refdoc=docname)
            ref.replace_self(new)


def setup(app):
    app.add_transform(PortableLinks)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
