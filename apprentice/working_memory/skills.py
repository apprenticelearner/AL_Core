from random import randint
import math

from apprentice.working_memory.adapters.experta_.factory import \
    ExpertaSkillFactory
from apprentice.working_memory.representation import Sai
from experta import Rule, Fact, W, KnowledgeEngine, MATCH, TEST, AS, NOT

max_depth = 1

fields = ["JCommTable.R0C0", "JCommTable.R1C0", "JCommTable2.R0C0",
          "JCommTable3.R0C0", "JCommTable3.R1C0", "JCommTable4.R0C0",
          "JCommTable4.R1C0", "JCommTable5.R0C0", "JCommTable5.R1C0",
          "JCommTable6.R0C0", "JCommTable6.R1C0", "JCommTable7.R0C0",
          "JCommTable8.R0C0"]
answer_field = ['JCommTable6.R0C0', 'JCommTable6.R1C0']

# RumbleBlocks fields
block_types = ["plat", "cube", "rect", "trap", "Checkpoint"]
pos_epsilon  = 0.2
pos_shift    = 0.1
rumble_print = True

def is_numeric_str(x):
    try:
        x = float(x)
        return True
    except Exception:
        return False

def rotate_clockwise(x):
    x = x + 90.0
    if x >= 360.0:
        x = x - 360.0
    return x


class FractionsEngine(KnowledgeEngine):

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn, Type=MATCH.bt, Level=MATCH.bl, 
            In_Inventory=False),
        TEST(lambda bt: bt in block_types),
        TEST(lambda lvl, bl: lvl == bl)
    )
    def remove_from_inventory(self, bn):
        if rumble_print:
            print("Removing " + str(bn) + " from inventory")
        return Sai(selection=bn,
                   action="Object_From_Inventory",
                   inputs={"Object":
                       {"Name": bn,
                        "Source": "SoarTech",
                        "Transform":
                            {"Position_X": 0.0,
                             "Position_Y": 0.0,
                             "Rotation": 0}}})

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn, Type=MATCH.bt, Level=MATCH.bl,
            Position_X=MATCH.x, Position_Y=MATCH.y, Rotation=MATCH.rot,
            Bounds_X=MATCH.xb, Bounds_Y=MATCH.yb,
            In_Inventory=False, On_Ground=MATCH.og),
        TEST(lambda bt: bt in block_types),
        TEST(lambda lvl, bl: lvl == bl)
    )
    def shift_left(self, bn, bt, bl, x, y, xb, yb, og, rot):
        if rumble_print:
            print("Shifting " + str(bn) + " to the left")
        self.declare(Fact(Name=bn,
                          Type=bt,
                          Level=bl,
                          Position_X=x,
                          Position_Y=y,
                          Bounds_X=(xb - pos_shift),
                          Bounds_Y=yb,
                          Rotation=rot,
                          In_Inventory=False,
                          On_Ground=og))

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn, Type=MATCH.bt, Level=MATCH.bl,
            Position_X=MATCH.x, Position_Y=MATCH.y, Rotation=MATCH.rot,
            Bounds_X=MATCH.xb, Bounds_Y=MATCH.yb,
            In_Inventory=False, On_Ground=MATCH.og),
        TEST(lambda bt: bt in block_types),
        TEST(lambda lvl, bl: lvl == bl)
    )
    def shift_right(self, bn, bt, bl, x, y, xb, yb, og, rot):
        if rumble_print:
            print("Shifting " + str(bn) + " to the right")
        self.declare(Fact(Name=bn,
                          Type=bt,
                          Level=bl,
                          Position_X=x,
                          Position_Y=y,
                          Bounds_X=(xb + pos_shift),
                          Bounds_Y=yb,
                          Rotation=rot,
                          In_Inventory=False,
                          On_Ground=og))

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn, Type=MATCH.bt, Level=MATCH.bl,
            Position_X=MATCH.x, Position_Y=MATCH.y, Rotation=MATCH.rot,
            Bounds_X=MATCH.xb, Bounds_Y=MATCH.yb,
            In_Inventory=False, On_Ground=False),
        TEST(lambda bt: bt in block_types),
        TEST(lambda lvl, bl: lvl == bl)
    )
    def shift_down(self, bn, bt, bl, x, y, xb, yb, rot):
        if rumble_print:
            print("Shifting " + str(bn) + " down")
        self.declare(Fact(Name=bn,
                          Type=bt,
                          Level=bl,
                          Position_X=x,
                          Position_Y=y,
                          Bounds_X=xb,
                          Bounds_Y=(yb - pos_shift),
                          Rotation=rot,
                          In_Inventory=False,
                          On_Ground=False))

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn, Type=MATCH.bt, Level=MATCH.bl,
            Position_X=MATCH.x, Position_Y=MATCH.y, Rotation=MATCH.rot,
            Bounds_X=MATCH.xb, Bounds_Y=MATCH.yb,
            In_Inventory=False),
        TEST(lambda bt: bt in block_types),
        TEST(lambda lvl, bl: lvl == bl)
    )
    def shift_up(self, bn, bt, bl, x, y, xb, yb, rot):
        if rumble_print:
            print("Shifting " + str(bn) + " to the left")
        self.declare(Fact(Name=bn,
                          Type=bt,
                          Level=bl,
                          Position_X=x,
                          Position_Y=y,
                          Bounds_X=xb,
                          Bounds_Y=(yb + pos_shift),
                          Rotation=rot,
                          In_Inventory=False,
                          On_Ground=False))

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn, Type=MATCH.bt, Level=MATCH.bl,
            In_Inventory=False,
            Position_X=MATCH.x, Position_Y=MATCH.y, Rotation=MATCH.rot),
        TEST(lambda bt: bt in block_types),
        TEST(lambda lvl, bl: lvl == bl)
    )
    def place_block(self, bn, x, y, rot):
        """
        Assuming internal logics have shifted the block's loation, places
        a block at whatever location it is at.
        """
        if rumble_print:
            print("Placing block at current location " + str(bn))
        return Sai(selection=bn,
                   action="Object_Released",
                   inputs={"Object":
                       {"Name": bn,
                        "Source": "SoarTech",
                        "Transform":
                            {"Position_X": x,
                             "Position_Y": y,
                             "Rotation": rot}}})

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn1, Type=MATCH.bt1, Level=MATCH.bl1,
            Bounds_Y=MATCH.yb1, Rotation=MATCH.rot,
            In_Inventory=False), # Block to be placed
        TEST(lambda bt1: bt1 in block_types),
        TEST(lambda lvl, bl1: lvl == bl1),
        Fact(Name=MATCH.bn2, Type=MATCH.bt2, Level=MATCH.bl2,
            Position_X=MATCH.x2, Position_Y=MATCH.y2,
            Bounds_Y=MATCH.yb2, In_Inventory=False), # Block that is present
        TEST(lambda bt2: bt2 in block_types),
        TEST(lambda lvl, bl2: lvl == bl2),
        TEST(lambda bn1, bn2: bn1 != bn2)
    )
    def place_block_above(self, bn1, yb1, rot, bn2, x2, y2, yb2):
        """
        Places a block directly above another one. The block information
        with "1" is the block to be placed above the block with "2."

        Bound arithmetic takes the bounds to be a diameter.
        """
        if rumble_print:
            print("Place block above called to place " + str(bn1)
                    + " on top of " + str(bn2))
        return Sai(selection=bn1,
                   action="Object_Released",
                   inputs={"Object":
                       {"Name": bn1,
                        "Source": "SoarTech",
                        "Transform":
                            {"Position_X": x2,
                             "Position_Y": y2 + ((yb1 + yb2) / 2.0),
                             "Rotation": rot}}})

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn1, Type=MATCH.bt1, Level=MATCH.bl1,
            Bounds_Y=MATCH.yb1, Rotation=MATCH.rot,
            In_Inventory=False), # Block to be placed
        TEST(lambda bt1: bt1 in block_types),
        TEST(lambda lvl, bl1: lvl == bl1),
        Fact(Name=MATCH.bn2, Type=MATCH.bt2, Level=MATCH.bl2,
            Position_X=MATCH.x2, Position_Y=MATCH.y2,
            Bounds_Y=MATCH.yb2, In_Inventory=False), # Block that is present
        TEST(lambda bt2: bt2 in block_types),
        TEST(lambda lvl, bl2: lvl == bl2),
        TEST(lambda bn1, bn2: bn1 != bn2)
    )
    def place_block_below(self, bn1, yb1, rot, bn2, x2, y2, yb2):
        """
        Places a block directly below another one. The block information
        with "1" is the block to be placed below the block with "2."

        Bound arithmetic takes the bounds to be a diameter.
        """
        if rumble_print:
            print("Place block below called to place " + str(bn1)
                    + " below " + str(bn2))
        return Sai(selection=bn1,
                   action="Object_Released",
                   inputs={"Object":
                       {"Name": bn1,
                        "Source": "SoarTech",
                        "Transform":
                            {"Position_X": x2,
                             "Position_Y": y2 - ((yb1 + yb2) / 2.0),
                             "Rotation": rot}}})

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn1, Type=MATCH.bt1, Level=MATCH.bl1,
            Bounds_X=MATCH.xb1, Rotation=MATCH.rot,
            In_Inventory=False), # Block to be placed
        TEST(lambda bt1: bt1 in block_types),
        TEST(lambda lvl, bl1: lvl == bl1),
        Fact(Name=MATCH.bn2, Type=MATCH.bt2, Level=MATCH.bl2,
            Position_X=MATCH.x2, Position_Y=MATCH.y2,
            Bounds_X=MATCH.xb2, In_Inventory=False), # Block that is present
        TEST(lambda bt2: bt2 in block_types),
        TEST(lambda lvl, bl2: lvl == bl2),
        TEST(lambda bn1, bn2: bn1 != bn2)
    )
    def place_block_left(self, bn1, xb1, rot, bn2, x2, y2, xb2):
        """
        Places a block directly to the left of another one. The block 
        information with "1" is the block to be placed to the left of the 
        block with "2."

        Bound arithmetic takes the bounds to be a diameter.
        """
        if rumble_print:
            print("Place block left called to place " + str(bn1)
                    + " to the left of " + str(bn2))
        return Sai(selection=bn1,
                   action="Object_Released",
                   inputs={"Object":
                       {"Name": bn1,
                        "Source": "SoarTech",
                        "Transform":
                            {"Position_X": x2 - ((xb1 + xb2) / 2.0),
                             "Position_Y": y2,
                             "Rotation": rot}}})

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn1, Type=MATCH.bt1, Level=MATCH.bl1,
            Bounds_X=MATCH.xb1, Rotation=MATCH.rot,
            In_Inventory=False), # Block to be placed
        TEST(lambda bt1: bt1 in block_types),
        TEST(lambda lvl, bl1: lvl == bl1),
        Fact(Name=MATCH.bn2, Type=MATCH.bt2, Level=MATCH.bl2,
            Position_X=MATCH.x2, Position_Y=MATCH.y2,
            Bounds_X=MATCH.xb2, In_Inventory=False), # Block that is present
        TEST(lambda bt2: bt2 in block_types),
        TEST(lambda lvl, bl2: lvl == bl2),
        TEST(lambda bn1, bn2: bn1 != bn2)
    )
    def place_block_right(self, bn1, xb1, rot, bn2, x2, y2, xb2):
        """
        Places a block directly to the right of another one. The block 
        information with "1" is the block to be placed to the right of the 
        block with "2."

        Bound arithmetic takes the bounds to be a diameter.
        """
        if rumble_print:
            print("Place block left called to place " + str(bn1)
                    + " to the right of " + str(bn2))
        return Sai(selection=bn1,
                   action="Object_Released",
                   inputs={"Object":
                       {"Name": bn1,
                        "Source": "SoarTech",
                        "Transform":
                            {"Position_X": x2 + ((xb1 + xb2) / 2.0),
                             "Position_Y": y2,
                             "Rotation": rot}}})

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn1, Type=MATCH.bt1, Level=MATCH.bl1,
            Rotation=MATCH.rot, In_Inventory=False), # Block to be placed
        TEST(lambda bt1: bt1 in block_types),
        TEST(lambda lvl, bl1: lvl == bl1),
        Fact(Name=MATCH.bn2, Type=MATCH.bt2, Level=MATCH.bl2,
            Position_X=MATCH.x2, Position_Y=MATCH.y2,
            In_Inventory=False), # Block 1
        TEST(lambda bt2: bt2 in block_types),
        TEST(lambda lvl, bl2: lvl == bl2),
        Fact(Name=MATCH.bn3, Type=MATCH.bt3, Level=MATCH.bl3,
            Position_X=MATCH.x3, Position_Y=MATCH.y3,
            In_Inventory=False), # Block 2
        TEST(lambda bt3: bt3 in block_types),
        TEST(lambda lvl, bl3: lvl == bl3),
        TEST(lambda bn1, bn2, bn3: bn1 != bn2 and bn2 != bn3 and bn1 != bn3),
        TEST(lambda y2, y3: abs(y2 - y3) <= pos_epsilon)
    )
    def place_block_between(self, bn1, rot, bn2, x2, y2, bn3, x3):
        """
        Places a block halfway between the x coordinates of two other blocks.
        Requires the two other blocks to be on the same level.

        Bound arithmetic takes the bounds to be a diameter.
        """
        if rumble_print:
            print("Place block between called to place " + str(bn1)
                    + " between " + str(bn2) + " and " + str(bn3))
        return Sai(selection=bn1,
                   action="Object_Released",
                   inputs={"Object":
                       {"Name": bn1,
                        "Source": "SoarTech",
                        "Transform":
                            {"Position_X": (x2 + x3) / 2.0,
                             "Position_Y": y2,
                             "Rotation": rot}}})

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.block_name, Type="ufo_block", Level=MATCH.bl),
        Fact(Type="Goal", Position_X=MATCH.xpos, Position_Y=MATCH.ypos,
            Level=MATCH.gl),
        TEST(lambda lvl, bl: lvl == bl),
        TEST(lambda lvl, gl: lvl == gl)
    )
    def place_ufo_on_goal(self, block_name, xpos, ypos):
        if rumble_print:
            print("Place UFO on goal called")
        return Sai(selection=block_name,
                   action="Object_Released",
                   inputs={"Object":
                       {"Name": block_name,
                        "Source": "SoarTech",
                        "Transform":
                            {"Position_X": xpos,
                             "Position_Y": ypos,
                             "Rotation": 0.0}}})

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn, Type=MATCH.bt, Level=MATCH.bl, 
            In_Inventory=False, Rotation=MATCH.rot),
        TEST(lambda bt: bt in block_types),
        TEST(lambda lvl, bl: lvl == bl),
        Fact(Type="Checkpoint", Level=MATCH.cl,
            Position_X=MATCH.xpos, Position_Y=MATCH.ypos),
        TEST(lambda lvl, cl: lvl == cl)
    )
    def place_block_on_checkpoint(self, bn, xpos, ypos, rot):
        if rumble_print:
            print("Place block on checkpoint called on " + str(bn))
        return Sai(selection=bn,
                   action="Object_Released",
                   inputs={"Object":
                       {"Name": bn,
                        "Source": "SoarTech",
                        "Transform":
                            {"Position_X": xpos,
                             "Position_Y": ypos,
                             "Rotation": rot}}})

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn, Type=MATCH.bt, Level=MATCH.bl,
            In_Inventory=False, Rotation=MATCH.rot),
        TEST(lambda bt: bt in block_types),
        TEST(lambda lvl, bl: lvl == bl),
        Fact(Name=MATCH.ck1, Type="Checkpoint", Level=MATCH.cl1,
            Position_X=MATCH.x1, Position_Y=MATCH.y1),  # Checkpoint 1
        Fact(Name=MATCH.ck2, Type="Checkpoint", Level=MATCH.cl2,
            Position_X=MATCH.x2, Position_Y=MATCH.y2),  # Checkpoint 2
        TEST(lambda ck1, ck2: ck1 != ck2),
        TEST(lambda lvl, cl1, cl2: lvl == cl1 and lvl == cl2),
        TEST(lambda y1, y2: abs(y1 - y2) <= pos_epsilon)
    )
    def place_block_between_checkpoints(self, bn, x1, y1, x2, rot):
        if rumble_print:
            print("Place block between checkpoints called on " + str(bn))
        return Sai(selection=bn,
                   action="Object_Released",
                   inputs={"Object":
                       {"Name": bn,
                        "Source": "SoarTech",
                        "Transform":
                            {"Position_X": (x1 + x2) / 2.0,
                             "Position_Y": y1,
                             "Rotation": rot}}})

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn, Type=MATCH.bt, Level=MATCH.bl, 
            In_Inventory=False,
            Bounds_Y=MATCH.yb, Rotation=MATCH.rot),
        TEST(lambda bt: bt in block_types),
        TEST(lambda lvl, bl: lvl == bl),
        Fact(Name="ground", Position_Y=MATCH.ground_y, Level=MATCH.gl),
        TEST(lambda lvl, gl: lvl == gl)
    )
    def place_block_on_ground(self, bn, yb, ground_y, rot):
        if rumble_print:
            print("Place block on ground called on " + str(bn))
        return Sai(selection=bn,
                   action="Object_Released",
                   inputs={"Object":
                       {"Name": bn,
                        "Source": "SoarTech",
                        "Transform":
                            {"Position_X": 0.0,
                             "Position_Y": ground_y + (yb / 2.0),
                             "Rotation": rot}}})

    @Rule(
        Fact(Curr_Level=MATCH.lvl),
        Fact(Name=MATCH.bn, Type=MATCH.bt, Level=MATCH.bl, 
            In_Inventory=False,
            Position_X=MATCH.x, Position_Y=MATCH.y,
            Bounds_X=MATCH.xb, Bounds_Y=MATCH.yb,
            Rotation=MATCH.rot, On_Ground=MATCH.og),
        TEST(lambda bt: bt in block_types and bt != "cube"),
        TEST(lambda lvl, bl: lvl == bl)
    )
    def rotate_block(self, bn, bt, bl, x, y, xb, yb, rot, og):
        if rumble_print:
            print("Rotate block called on " + str(bn))
        self.declare(Fact(Name=bn,
                          Type=bt,
                          Level=bl,
                          Position_X=x,
                          Position_Y=y,
                          Bounds_X=yb,
                          Bounds_Y=xb,
                          Rotation=rotate_clockwise(rot),
                          In_Inventory=False,
                          On_Ground=og))

###########################################################################

    @Rule(
        Fact(id='done')
    )
    def click_done(self):
        # print('clicking done')
        return Sai(selection='done',
                   action='ButtonPressed',
                   inputs={'value': -1})
        # inputs={'value': '-1'})

    @Rule(
        Fact(id="JCommTable8.R0C0", contentEditable=True, value="")
    )
    def check(self):
        # print('checking box')
        return Sai(selection="JCommTable8.R0C0",
                   action='UpdateTextArea',
                   inputs={'value': "x"})

    @Rule(
        Fact(id=MATCH.id1, contentEditable=False, value=MATCH.value1),
        TEST(lambda id1, value1: id1 in fields and value1 != ""),
        Fact(id=MATCH.id2, contentEditable=False, value=MATCH.value2),
        TEST(lambda id2, value2: id2 in fields and value2 != ""),
        TEST(lambda id1, id2: id1 < id2),
        NOT(Fact(relation='equal', ele1=MATCH.id1, ele2=MATCH.id2))
    )
    def equal(self, id1, value1, id2, value2):
        new_id = "equal(%s, %s)" % (id1, id2)
        equality = value1 == value2
        # print('declaring equality', id1, id2, equality)
        self.declare(Fact(id=new_id,
                          relation='equal',
                          ele1=id1,
                          ele2=id2,
                          r_val=equality))

    @Rule(
        Fact(id='JCommTable8.R0C0', contentEditable=False, value='x'),
        Fact(id=W(), contentEditable=False, value=MATCH.value),
        TEST(lambda value: value != "" and is_numeric_str(value)),
        Fact(id=MATCH.field_id, contentEditable=True, value=W()),
        TEST(lambda field_id: field_id != 'JCommTable8.R0C0' and field_id not
                              in answer_field),
    )
    def update_convert_field(self, field_id, value):
        # print('updating convert field', field_id, value)
        return Sai(selection=field_id,
                   # action='UpdateTextField',
                   action='UpdateTextArea',
                   inputs={'value': value})

    @Rule(
        Fact(id=W(), contentEditable=False, value=MATCH.value),
        TEST(lambda value: value != "" and is_numeric_str(value)),
        Fact(id=MATCH.field_id, contentEditable=True, value=W()),
        TEST(lambda field_id: field_id != 'JCommTable8.R0C0'),
        TEST(lambda field_id: field_id in answer_field)
    )
    def update_answer_field(self, field_id, value):
        # field_id : str
        # print('updating answer field', field_id, value)
        return Sai(selection=field_id,
                   # action='UpdateTextField',
                   action='UpdateTextArea',
                   inputs={'value': value})

    @Rule(
        AS.fact1 << Fact(id=MATCH.id1, contentEditable=False,
                         value=MATCH.value1),
        TEST(lambda fact1: 'depth' not in fact1 or fact1['depth'] < max_depth),
        TEST(lambda value1: is_numeric_str(value1)),
        AS.fact2 << Fact(id=MATCH.id2, contentEditable=False,
                         value=MATCH.value2),
        TEST(lambda id1, id2: id1 <= id2),
        TEST(lambda fact2: 'depth' not in fact2 or fact2['depth'] < max_depth),
        TEST(lambda value2: is_numeric_str(value2)),
        NOT(Fact(operator='add', ele1=MATCH.id1, ele2=MATCH.id2))
    )
    def add(self, id1, value1, fact1, id2, value2, fact2):
        new_id = 'add(%s, %s)' % (id1, id2)

        new_value = float(value1) + float(value2)
        if new_value.is_integer():
            new_value = int(new_value)
        new_value = str(new_value)

        depth1 = 0 if 'depth' not in fact1 else fact1['depth']
        depth2 = 0 if 'depth' not in fact2 else fact2['depth']
        new_depth = 1 + max(depth1, depth2)

        # print('adding', id1, id2)

        # TODO define a declare that takes an object and returns a target
        self.declare(Fact(id=new_id,
                          operator='add',
                          ele1=id1,
                          ele2=id2,
                          contentEditable=False,
                          value=new_value,
                          depth=new_depth))

    @Rule(
        AS.fact1 << Fact(id=MATCH.id1, contentEditable=False,
                         value=MATCH.value1),
        TEST(lambda fact1: 'depth' not in fact1 or fact1['depth'] < max_depth),
        TEST(lambda value1: is_numeric_str(value1)),
        AS.fact2 << Fact(id=MATCH.id2, contentEditable=False,
                         value=MATCH.value2),
        TEST(lambda id1, id2: id1 <= id2),
        TEST(lambda fact2: 'depth' not in fact2 or fact2['depth'] < max_depth),
        TEST(lambda value2: is_numeric_str(value2)),
        NOT(Fact(operator='multiply', ele1=MATCH.id1, ele2=MATCH.id2))
    )
    def multiply(self, id1, value1, fact1, id2, value2, fact2):
        # print('multiplying', id1, id2)
        new_id = 'multiply(%s, %s)' % (id1, id2)

        new_value = float(value1) * float(value2)
        if new_value.is_integer():
            new_value = int(new_value)
        new_value = str(new_value)

        depth1 = 0 if 'depth' not in fact1 else fact1['depth']
        depth2 = 0 if 'depth' not in fact2 else fact2['depth']
        new_depth = 1 + max(depth1, depth2)

        self.declare(Fact(id=new_id,
                          operator='multiply',
                          ele1=id1,
                          ele2=id2,
                          contentEditable=False,
                          value=new_value,
                          depth=new_depth))

    @Rule(
      AS.fact1 << Fact(id=MATCH.id1, contentEditable=False,
                       value=MATCH.value1),
      TEST(lambda fact1: 'depth' not in fact1 or fact1['depth'] < max_depth),
      TEST(lambda value1: is_numeric_str(value1)),
      AS.fact2 << Fact(id=MATCH.id2, contentEditable=False,
                       value=MATCH.value2),
      TEST(lambda id1, id2: id1 <= id2),
      TEST(lambda fact2: 'depth' not in fact2 or fact2['depth'] < max_depth),
      TEST(lambda value2: is_numeric_str(value2)),
      NOT(Fact(operator='lcm', ele1=MATCH.id1, ele2=MATCH.id2))
    )
    def least_common_multiple(self, id1, value1, fact1, id2, value2, fact2):
        new_id = ' lcm({0}, {1})'.format(id1, id2)

        gcd = math.gcd(int(value1), int(value2))
        new_value = abs(int(value1) * int(value2)) // gcd
        # if new_value.is_integer():
        #     new_value = int(new_value)
        new_value = str(new_value)

        depth1 = 0 if 'depth' not in fact1 else fact1['depth']
        depth2 = 0 if 'depth' not in fact2 else fact2['depth']
        new_depth = 1 + max(depth1, depth2)

        self.declare(Fact(id=new_id,
                          operator='lcm',
                          ele1=id1,
                          ele2=id2,
                          contentEditable=False,
                          value=new_value,
                          depth=new_depth))

    @Rule(
        Fact(id='JCommTable.R0C0', contentEditable=False, value=MATCH.value1),
        Fact(id='JCommTable2.R0C0', contentEditable=False, value="*"),
        Fact(id='JCommTable3.R0C0', contentEditable=False, value=MATCH.value2),
        Fact(id='JCommTable6.R0C0', contentEditable=True)
    )
    def correct_multiply_num(self, value1, value2):
        new_value = float(value1) * float(value2)
        if new_value.is_integer():
            new_value = int(new_value)
        new_value = str(new_value)

        return Sai(selection='JCommTable6.R0C0',
                   # action='UpdateTextField',
                   action='UpdateTextArea',
                   inputs={'value': new_value})

    @Rule(
        Fact(id='JCommTable.R1C0', contentEditable=False, value=MATCH.value1),
        Fact(id='JCommTable2.R0C0', contentEditable=False, value="*"),
        Fact(id='JCommTable3.R1C0', contentEditable=False, value=MATCH.value2),
        Fact(id='JCommTable6.R1C0', contentEditable=True)
    )
    def correct_multiply_denom(self, value1, value2):
        new_value = float(value1) * float(value2)
        if new_value.is_integer():
            new_value = int(new_value)
        new_value = str(new_value)

        return Sai(selection='JCommTable6.R1C0',
                   # action='UpdateTextField',
                   action='UpdateTextArea',
                   inputs={'value': new_value})

    @Rule(
        Fact(id='JCommTable6.R0C0', contentEditable=False),
        Fact(id='JCommTable6.R1C0', contentEditable=False),
        Fact(id='done')
    )
    def correct_done(self):
        return Sai(selection='done',
                   action='ButtonPressed',
                   inputs={'value': -1})

    @Rule(
        Fact(id='JCommTable.R0C0', contentEditable=False, value=MATCH.value1),
        Fact(id='JCommTable2.R0C0', contentEditable=False, value="+"),
        Fact(id='JCommTable3.R0C0', contentEditable=False, value=MATCH.value2),
        Fact(id='JCommTable.R1C0', contentEditable=False, value=MATCH.value3),
        Fact(id='JCommTable3.R1C0', contentEditable=False, value=MATCH.value3),
        Fact(id='JCommTable6.R0C0', contentEditable=True)
    )
    def correct_add_same_num(self, value1, value2):
        new_value = float(value1) + float(value2)
        if new_value.is_integer():
            new_value = int(new_value)
        new_value = str(new_value)

        return Sai(selection='JCommTable6.R0C0',
                   # action='UpdateTextField',
                   action='UpdateTextArea',
                   inputs={'value': new_value})

    @Rule(
        Fact(id='JCommTable.R0C0', contentEditable=False, value=MATCH.value1),
        Fact(id='JCommTable2.R0C0', contentEditable=False, value="+"),
        Fact(id='JCommTable3.R0C0', contentEditable=False, value=MATCH.value2),
        Fact(id='JCommTable.R1C0', contentEditable=False, value=MATCH.value3),
        Fact(id='JCommTable3.R1C0', contentEditable=False, value=MATCH.value3),
        Fact(id='JCommTable6.R1C0', contentEditable=True)
    )
    def correct_copy_same_denom(self, value3):
        return Sai(selection='JCommTable6.R1C0',
                   # action='UpdateTextField',
                   action='UpdateTextArea',
                   inputs={'value': value3})

    @Rule(
        Fact(id="JCommTable.R1C0", contentEditable=False, value=MATCH.denom1),
        Fact(id="JCommTable2.R0C0", contentEditable=False, value="+"),
        Fact(id="JCommTable3.R1C0", contentEditable=False, value=MATCH.denom2),
        TEST(lambda denom1, denom2: denom1 != denom2),
        Fact(id="JCommTable8.R0C0", contentEditable=True, value="")
    )
    def correct_check(self):
        # print('checking box')
        return Sai(selection="JCommTable8.R0C0",
                   action='UpdateTextArea',
                   inputs={'value': "x"})

    @Rule(
        Fact(id='JCommTable2.R0C0', contentEditable=False, value="+"),
        Fact(id="JCommTable8.R0C0", contentEditable=False, value="x"),
        Fact(id='JCommTable.R0C0', contentEditable=False, value=MATCH.value1),
        Fact(id='JCommTable3.R1C0', contentEditable=False, value=MATCH.value2),
        Fact(id='JCommTable4.R1C0', contentEditable=False),
        Fact(id='JCommTable4.R0C0', contentEditable=True)
    )
    def correct_convert_num1(self, value1, value2):
        new_value = float(value1) * float(value2)
        if new_value.is_integer():
            new_value = int(new_value)
        new_value = str(new_value)

        return Sai(selection='JCommTable4.R0C0',
                   # action='UpdateTextField',
                   action='UpdateTextArea',
                   inputs={'value': new_value})

    @Rule(
        Fact(id='JCommTable2.R0C0', contentEditable=False, value="+"),
        Fact(id="JCommTable8.R0C0", contentEditable=False, value="x"),
        Fact(id='JCommTable.R1C0', contentEditable=False, value=MATCH.value1),
        Fact(id='JCommTable3.R0C0', contentEditable=False, value=MATCH.value2),
        Fact(id='JCommTable4.R0C0', contentEditable=False),
        Fact(id='JCommTable4.R1C0', contentEditable=False),
        Fact(id='JCommTable5.R1C0', contentEditable=False),
        Fact(id='JCommTable5.R0C0', contentEditable=True)
    )
    def correct_convert_num2(self, value1, value2):
        new_value = float(value1) * float(value2)
        if new_value.is_integer():
            new_value = int(new_value)
        new_value = str(new_value)

        return Sai(selection='JCommTable5.R0C0',
                   # action='UpdateTextField',
                   action='UpdateTextArea',
                   inputs={'value': new_value})

    @Rule(
        Fact(id='JCommTable2.R0C0', contentEditable=False, value="+"),
        Fact(id="JCommTable8.R0C0", contentEditable=False, value="x"),
        Fact(id='JCommTable.R1C0', contentEditable=False, value=MATCH.value1),
        Fact(id='JCommTable3.R1C0', contentEditable=False, value=MATCH.value2),
        Fact(id='JCommTable4.R1C0', contentEditable=True)
    )
    def correct_convert_denom1(self, value1, value2):
        new_value = float(value1) * float(value2)
        if new_value.is_integer():
            new_value = int(new_value)
        new_value = str(new_value)

        return Sai(selection='JCommTable4.R1C0',
                   # action='UpdateTextField',
                   action='UpdateTextArea',
                   inputs={'value': new_value})

    @Rule(
        Fact(id='JCommTable2.R0C0', contentEditable=False, value="+"),
        Fact(id="JCommTable8.R0C0", contentEditable=False, value="x"),
        Fact(id='JCommTable.R1C0', contentEditable=False, value=MATCH.value1),
        Fact(id='JCommTable3.R1C0', contentEditable=False, value=MATCH.value2),
        Fact(id='JCommTable4.R1C0', contentEditable=False),
        Fact(id='JCommTable5.R1C0', contentEditable=True)
    )
    def correct_convert_denom2(self, value1, value2):
        new_value = float(value1) * float(value2)
        if new_value.is_integer():
            new_value = int(new_value)
        new_value = str(new_value)

        return Sai(selection='JCommTable5.R1C0',
                   # action='UpdateTextField',
                   action='UpdateTextArea',
                   inputs={'value': new_value})

    @Rule(
        Fact(id='JCommTable4.R0C0', contentEditable=False, value=MATCH.value1),
        Fact(id='JCommTable4.R1C0', contentEditable=False, value=MATCH.value3),
        Fact(id='JCommTable7.R0C0', contentEditable=False, value="+"),
        Fact(id='JCommTable5.R0C0', contentEditable=False, value=MATCH.value2),
        Fact(id='JCommTable5.R1C0', contentEditable=False, value=MATCH.value3),
        Fact(id='JCommTable6.R0C0', contentEditable=True),
    )
    def correct_add_convert_num(self, value1, value2):
        new_value = float(value1) + float(value2)
        if new_value.is_integer():
            new_value = int(new_value)
        new_value = str(new_value)

        return Sai(selection='JCommTable6.R0C0',
                   # action='UpdateTextField',
                   action='UpdateTextArea',
                   inputs={'value': new_value})

    @Rule(
        Fact(id='JCommTable4.R0C0', contentEditable=False, value=MATCH.value1),
        Fact(id='JCommTable4.R1C0', contentEditable=False, value=MATCH.value3),
        Fact(id='JCommTable7.R0C0', contentEditable=False, value="+"),
        Fact(id='JCommTable5.R0C0', contentEditable=False, value=MATCH.value2),
        Fact(id='JCommTable5.R1C0', contentEditable=False, value=MATCH.value3),
        Fact(id='JCommTable6.R1C0', contentEditable=True),
    )
    def correct_copy_convert_denom(self, value3):
        return Sai(selection='JCommTable6.R1C0',
                   # action='UpdateTextField',
                   action='UpdateTextArea',
                   inputs={'value': value3})

# Use class above, comment out skills, and define own skills for RumbleBlocks
# Alternatively, leave what's there alone, add new ones for RB
ke = FractionsEngine()
skill_factory = ExpertaSkillFactory(ke)
click_done_skill = skill_factory.from_ex_rule(ke.click_done)
check_skill = skill_factory.from_ex_rule(ke.check)
equal_skill = skill_factory.from_ex_rule(ke.equal)
update_answer_field_skill = skill_factory.from_ex_rule(ke.update_answer_field)
update_convert_field_skill = skill_factory.from_ex_rule(
    ke.update_convert_field)
add_skill = skill_factory.from_ex_rule(ke.add)
multiply_skill = skill_factory.from_ex_rule(ke.multiply)
least_common_multiple = skill_factory.from_ex_rule(ke.least_common_multiple)

correct_multiply_num = skill_factory.from_ex_rule(ke.correct_multiply_num)
correct_multiply_denom = skill_factory.from_ex_rule(ke.correct_multiply_denom)

correct_add_same_num = skill_factory.from_ex_rule(ke.correct_add_same_num)
correct_copy_same_denom = skill_factory.from_ex_rule(
    ke.correct_copy_same_denom)

correct_check = skill_factory.from_ex_rule(ke.correct_check)
correct_convert_num1 = skill_factory.from_ex_rule(ke.correct_convert_num1)
correct_convert_num2 = skill_factory.from_ex_rule(ke.correct_convert_num2)
correct_convert_denom1 = skill_factory.from_ex_rule(ke.correct_convert_denom1)
correct_convert_denom2 = skill_factory.from_ex_rule(ke.correct_convert_denom2)
correct_add_convert_num = skill_factory.from_ex_rule(
    ke.correct_add_convert_num)
correct_copy_convert_denom = skill_factory.from_ex_rule(
    ke.correct_copy_convert_denom)

correct_done = skill_factory.from_ex_rule(ke.correct_done)

# RumbleBlocks skill factory extraction
shift_left          = skill_factory.from_ex_rule(ke.shift_left)
shift_right         = skill_factory.from_ex_rule(ke.shift_right)
shift_up            = skill_factory.from_ex_rule(ke.shift_up)
shift_down          = skill_factory.from_ex_rule(ke.shift_down)
place_block         = skill_factory.from_ex_rule(ke.place_block)
place_block_above   = skill_factory.from_ex_rule(ke.place_block_above)
place_block_below   = skill_factory.from_ex_rule(ke.place_block_below)
place_block_left    = skill_factory.from_ex_rule(ke.place_block_left)
place_block_right   = skill_factory.from_ex_rule(ke.place_block_right)
place_block_between = skill_factory.from_ex_rule(ke.place_block_between)
place_ufo_on_goal   = skill_factory.from_ex_rule(ke.place_ufo_on_goal)
place_block_on_checkpoint = skill_factory.from_ex_rule(ke.place_block_on_checkpoint)
place_block_between_checkpoints = skill_factory.from_ex_rule(ke.place_block_between_checkpoints)
place_block_on_ground = skill_factory.from_ex_rule(ke.place_block_on_ground)
rotate_block        = skill_factory.from_ex_rule(ke.rotate_block)
remove_from_inventory = skill_factory.from_ex_rule(ke.remove_from_inventory)

fraction_skill_set = {'click_done': click_done_skill, 'check': check_skill,
                      'update_answer': update_answer_field_skill,
                      'update_convert': update_convert_field_skill,
                      'equal': equal_skill,
                      'add': add_skill,
                      'multiply': multiply_skill,
                      'least_common_multiple': least_common_multiple,

                      'correct_multiply_num': correct_multiply_num,
                      'correct_multiply_denom': correct_multiply_denom,
                      'correct_done': correct_done,
                      'correct_add_same_num': correct_add_same_num,
                      'correct_copy_same_denom': correct_copy_same_denom,
                      'correct_check': correct_check,
                      'correct_convert_num1': correct_convert_num1,
                      'correct_convert_num2': correct_convert_num2,
                      'correct_convert_denom1': correct_convert_denom1,
                      'correct_convert_denom2': correct_convert_denom2,
                      'correct_add_convert_num': correct_add_convert_num,
                      'correct_copy_convert_denom': correct_copy_convert_denom,

                      'remove_from_inventory': remove_from_inventory,
                      'shift_left': shift_left,
                      'shift_right': shift_right,
                      'shift_up': shift_up,
                      'shift_down': shift_down,
                      'place_block': place_block,
                      'place_block_above': place_block_above,
                      'place_block_below': place_block_below,
                      'place_block_left': place_block_left,
                      'place_block_right': place_block_right,
                      'place_block_between': place_block_between,
                      'place_ufo_on_goal': place_ufo_on_goal,
                      'place_block_on_checkpoint': place_block_on_checkpoint,
                      'place_block_between_checkpoints': place_block_between_checkpoints,
                      'place_block_on_ground': place_block_on_ground,
                      'rotate_block': rotate_block,
                      }


class RandomFracEngine(KnowledgeEngine):
    @Rule(
        Fact(id=MATCH.id, contentEditable=True, value=W())
    )
    def input_random(self, id):
        return Sai(selection=id, action='UpdateTextArea',
                   inputs={'value': str(randint(0, 100))})

    @Rule(
        Fact(id='done')
    )
    def click_done(self):
        return Sai(selection='done', action='ButtonPressed',
                   inputs={'value': -1})


def fact_from_dict(f):
    if '__class__' in f:
        fact_class = f['__class__']
    else:
        fact_class = Fact
    f2 = {k: v for k, v in f.items() if k[:2] != "__"}
    return fact_class(f2)


if __name__ == "__main__":
    from apprentice.working_memory import ExpertaWorkingMemory
    import copy

    # c = copy.deepcopy(fraction_skill_set['click_done'])
    # prior_skills = [fraction_skill_set['click_done']]
    prior_skills = None
    # wm = ExpertaWorkingMemory(ke=KnowledgeEngine())
    # wm.add_skills(prior_skills)
    # import collections.OrderedDict
    if prior_skills is None:
        prior_skills = {
            "click_done": False,  # True,
            "check": False,  # True,
            "equal": False,
            "update_answer": False, # True,
            "update_convert": False,  # , True,
            "add": False,  # True,
            "multiply": False,  # , True,

            # RumbleBlocks skills
            "remove_from_inventory": True,
            "shift_left": True,
            "shift_right": True,
            "shift_down": True,
            "shift_up": True,
            "place_block": True,
            "place_block_above": True,
            "place_block_below": True,
            "place_block_left": True,
            "place_block_right": True,
            "place_block_between": True,
            "place_ufo_on_goal": True,
            "place_block_on_checkpoint": True,
            "place_block_between_checkpoints": True,
            "place_block_on_ground": True,
            "rotate_block": True,
        }

    wm = ExpertaWorkingMemory(ke=KnowledgeEngine())

    skill_map = fraction_skill_set
    prior_skills = [
        skill_map[s]
        for s, active in prior_skills.items()
        if active and s in skill_map
    ]
    wm.add_skills(prior_skills)

    temp = wm.ke.matcher
    # wm.ke.matcher = None

    c = copy.deepcopy(wm)
    wm.ke.matcher = temp

    # self.ke.matcher.__init__(self.ke)
    # self.ke.reset()
    # ftn = wm.ke.matcher.root_node.children[0]
    # cb = ftn.callback
    # copy.deepcopy(cb)
