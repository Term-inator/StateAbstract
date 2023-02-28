import functools
import xml
from enum import Enum
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np

import force_directed_layout


def pretty_xml(element: ET.Element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def _cmp(a, b):
    _map = {
        NTA: '1000',

        Declaration: '0100',
        Template: '0200',
        System: '0300',

        Name: '0011',
        Location: '0020',
        Init: '0030',
        BranchPoint: '0040',
        Transition: '0050',

        Source: '0002',
        Target: '0003',
        Label: '0004',
        Nail: '0005',
    }

    if a.depth != b.depth:
        return a.depth - b.depth
    return int(_map[type(a)][a.depth]) - int(_map[type(b)][b.depth])


class XMLNode:
    parent = None
    tag: str
    attrib: dict
    text: Optional[str]
    children: list
    depth: int = 0

    def __init__(self, parent, tag):
        self.parent = parent
        if parent is not None:
            parent.add_child(self)
            self.depth = parent.depth + 1

        self.tag = tag
        self.attrib = {}
        self.text = None
        self.children = []

    def add_attrib(self, key, value):
        self.attrib[key] = value
        return self

    def set_text(self, text):
        self.text = text
        return self

    def add_child(self, xml_node):
        self.children.append(xml_node)
        return self

    def to_xml(self):
        if hasattr(self, 'get_attrib'):
            self.attrib.update(self.get_attrib())
        xml_node = ET.Element(self.tag, attrib=self.attrib)
        if self.text is not None:
            xml_node.text = self.text
        self.children.sort(key=functools.cmp_to_key(_cmp))
        for child in self.children:
            xml_node.append(child.to_xml())
        return xml_node


class Attribute:
    _attrib: dict

    def __init__(self):
        self._attrib = {}

    def add_attrib(self, key, value):
        self._attrib[key] = value
        return self

    def get_attrib(self):
        return self._attrib


class IdAttribute(Attribute):
    def __init__(self):
        super().__init__()

    def set_id(self, id):
        self.add_attrib('id', id)
        return self


class CoordinateAttribute(Attribute):
    def __init__(self):
        super().__init__()

    def set_x(self, x):
        self.add_attrib('x', x)
        return self

    def set_y(self, y):
        self.add_attrib('y', y)
        return self

    def set_coordinate(self, x, y):
        self.set_x(x)
        self.set_y(y)
        return self


class RefAttribute(Attribute):
    def __init__(self):
        super().__init__()

    def set_ref(self, ref):
        self.add_attrib('ref', ref)
        return self


class Name(XMLNode, CoordinateAttribute):
    def __init__(self, parent: XMLNode):
        super().__init__(parent, 'name')
        CoordinateAttribute.__init__(self)


class LabelKindEnum(Enum):
    GUARD = "guard"
    ASSIGNMENT = "assignment"
    SYNCHRONISATION = "synchronisation"
    INVARIANT = "invariant"
    SELECT = "select"
    PROBABILITY = "probability"


class Label(XMLNode, CoordinateAttribute):
    def __init__(self, parent: XMLNode):
        super().__init__(parent, 'label')
        CoordinateAttribute.__init__(self)

    def set_kind(self, kind: LabelKindEnum):
        self.add_attrib('kind', kind.value)
        return self


class Init(XMLNode, RefAttribute):
    def __init__(self, parent: XMLNode):
        super().__init__(parent, 'init')
        RefAttribute.__init__(self)


class Source(XMLNode, RefAttribute):
    def __init__(self, parent: XMLNode):
        super().__init__(parent, 'source')
        RefAttribute.__init__(self)


class Target(XMLNode, RefAttribute):
    def __init__(self, parent: XMLNode):
        super().__init__(parent, 'target')
        RefAttribute.__init__(self)


class Nail(XMLNode, CoordinateAttribute):
    def __init__(self, parent: XMLNode):
        super().__init__(parent, 'nail')
        CoordinateAttribute.__init__(self)


class Declaration(XMLNode, IdAttribute):
    def __init__(self, parent: XMLNode):
        super().__init__(parent, 'declaration')
        IdAttribute.__init__(self)


class Location(XMLNode, IdAttribute, CoordinateAttribute):
    def __init__(self, parent: XMLNode):
        super().__init__(parent, 'location')
        IdAttribute.__init__(self)
        CoordinateAttribute.__init__(self)


class BranchPoint(XMLNode, IdAttribute, CoordinateAttribute):
    def __init__(self, parent: XMLNode):
        super().__init__(parent, 'branchpoint')
        IdAttribute.__init__(self)
        CoordinateAttribute.__init__(self)


class Transition(XMLNode):
    def __init__(self, parent: XMLNode):
        super().__init__(parent, 'transition')


class Template(XMLNode):
    def __init__(self, parent: XMLNode):
        super().__init__(parent, 'template')


class System(XMLNode):
    def __init__(self, parent: XMLNode):
        super().__init__(parent, 'system')


class NTA(XMLNode):
    def __init__(self, parent: Optional[XMLNode]):
        super().__init__(parent, 'nta')


class UppaalParser:
    def __init__(self, graph):
        self.root = NTA(None)
        self.graph = graph

        self.parse()

    def parse(self):
        Declaration(self.root)
        template = Template(self.root)
        Name(template).set_text('Ego')
        self.parse_locations(template)

        Init(template).set_ref('state0')

        self.parse_transitions(template)

        System(self.root).set_text('system Ego;')

    def parse_locations(self, template):
        layout = force_directed_layout.ForceDirectedLayout(self.graph)
        self.pos = layout.run(iterations=200)

        for key in self.graph.nodes:
            location = Location(template)
            location.set_id(f'state{key}').set_coordinate(f'{self.pos[key][0]:.2f}', f'{self.pos[key][1]:.2f}')
            Name(location).set_coordinate(f'{self.pos[key][0]-10:.2f}', f'{self.pos[key][1]-10:.2f}').set_text(f'state{key}')

    def parse_transitions(self, template):
        for key in self.graph.nodes:
            branch_point_group = {}  # action -> [(tag, weight)]
            branch_point_pos = {}  # action -> [x, y]
            if len(self.graph.nodes[key].children) > 1:
                for tag, action in self.graph.nodes[key].children:
                    if action not in branch_point_group:
                        branch_point_group[action] = []
                        branch_point_pos[action] = np.array([0, 0], dtype=np.float64)
                    weight = self.graph.nodes[key].children[(tag, action)]
                    branch_point_group[action].append((tag, weight))
                    print(self.graph.nodes[tag].state.coordinate)
                    branch_point_pos[action] += self.pos[tag]

            print(branch_point_group)
            branchpoint_index = 0
            for action in branch_point_group:
                group_size = len(branch_point_group[action])

                if group_size > 1:
                    branch_point = BranchPoint(template)
                    branch_point_coordinate = branch_point_pos[action] / group_size
                    branch_point.set_id(f'branch{key}_{branchpoint_index}').set_coordinate(f'{branch_point_coordinate[0]:.2f}',
                                                                       f'{branch_point_coordinate[1]:.2f}').set_text('1')

                    transition = Transition(template)
                    Source(transition).set_ref(f'state{key}')
                    Target(transition).set_ref(f'branch{key}_{branchpoint_index}')
                    Label(transition).set_kind(LabelKindEnum.GUARD).set_text(f'1 == {action[0]} && 2 == {action[1]}')

                    for tag, weight in branch_point_group[action]:
                        transition = Transition(template)
                        Source(transition).set_ref(f'branch{key}_{branchpoint_index}')
                        Target(transition).set_ref(f'state{tag}')
                        Label(transition).set_kind(LabelKindEnum.PROBABILITY).set_text(str(weight))

                    branchpoint_index += 1
                else:
                    for tag, action in self.graph.nodes[key].children:
                        transition = Transition(template)
                        Source(transition).set_ref(f'state{key}')
                        Target(transition).set_ref(f'state{tag}')
                        Label(transition).set_kind(LabelKindEnum.GUARD).set_text(
                            f'1 == {action[0]} && 2 == {action[1]}')
            else:
                for tag, action in self.graph.nodes[key].children:
                    transition = Transition(template)
                    Source(transition).set_ref(f'state{key}')
                    Target(transition).set_ref(f'state{tag}')
                    Label(transition).set_kind(LabelKindEnum.GUARD).set_text(f'1 == {action[0]} && 2 == {action[1]}')

    def to_xml(self):
        return ET.ElementTree(self.root.to_xml())


def write(tree, filename):
    pretty_xml(tree.getroot(), indent='  ', newline='\n')
    text = ET.tostring(tree.getroot(), encoding='unicode')
    head = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.1//EN' 'http://www.it.uu.se/research/group/darts/uppaal/flat-1_2.dtd'>
"""
    text = head + text
    with open(f'{filename}.xml', 'w', encoding='utf-8') as f:
        f.write(text)
